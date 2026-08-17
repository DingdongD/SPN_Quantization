import unittest
from types import SimpleNamespace

from scripts import run_nyu_cspn_task_sensitive_bits as runner
from spn_quant import cspn_task_sensitive_bits as allocation


def ordered_union(sequences):
    values = []
    for sequence in sequences:
        for value in sequence:
            if value not in values:
                values.append(value)
    return tuple(values)


def registry():
    modules = ordered_union(
        allocation.WEIGHT_MODULES_BY_BLOCK[block]
        for block in allocation.BLOCK_ORDER)
    owners = ordered_union(
        allocation.ACTIVATION_OWNERS_BY_BLOCK[block]
        for block in allocation.BLOCK_ORDER)
    return allocation.build_registry(modules, owners)


def assignment():
    current = registry()
    modules = ordered_union(current.weights_by_block.values())
    owners = ordered_union(current.activations_by_block.values())
    return allocation.BitAssignment(
        weight_bits=tuple(
            (module, allocation.BIT_OPTIONS[index % 4])
            for index, module in enumerate(modules)),
        activation_bits=tuple(
            (owner, allocation.BIT_OPTIONS[index % 4])
            for index, owner in enumerate(owners)),
    )


def candidate():
    return runner.RuntimeCandidate("MIXED", "joint", assignment())


class RuntimeConfigurationTest(unittest.TestCase):
    def test_runtime_configuration_is_complete_and_keeps_propagation_fixed(self):
        current = candidate()

        config = runner.runtime_configuration(current)

        expected_weights = dict(current.assignment.weight_bits)
        expected_activations = dict(current.assignment.activation_bits)
        del expected_weights[runner.STEM_WEIGHT_MODULE]
        del expected_activations[runner.STEM_INPUT_OWNER]
        self.assertEqual(config["w_bits"], 4)
        self.assertEqual(config["a_bits"], 4)
        self.assertEqual(
            dict(config["weight_bit_overrides"]), expected_weights)
        self.assertEqual(
            dict(config["activation_bit_overrides"]),
            expected_activations)
        self.assertEqual(
            config["propagation"], runner.base.PROPAGATION_A8_Q13)
        self.assertFalse(config["quantize_bias"])
        self.assertEqual(config["group_size"], 8)

    def test_runtime_configuration_rejects_incomplete_and_forbidden_sites(self):
        current = candidate()
        weights = current.assignment.weight_bits
        activations = current.assignment.activation_bits
        missing = runner.RuntimeCandidate(
            "MISSING", "joint",
            allocation.BitAssignment(weights[:-1], activations))
        forbidden = runner.RuntimeCandidate(
            "FORBIDDEN", "joint",
            allocation.BitAssignment(
                weights,
                activations + ((('guidance', 'confidence'), 4),)))

        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            runner.runtime_configuration(missing)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            runner.runtime_configuration(forbidden)

    def test_validate_configured_precision_checks_every_site_and_stem(self):
        current = candidate()
        expected_weights = dict(current.assignment.weight_bits)
        expected_activations = dict(current.assignment.activation_bits)
        stem_weight_bits = expected_weights.pop(runner.STEM_WEIGHT_MODULE)
        stem_activation_bits = expected_activations.pop(
            runner.STEM_INPUT_OWNER)
        rotation_owners = {
            owner for owner in expected_activations
            if owner[0].startswith("rotation.")}
        ordinary = {
            owner: SimpleNamespace(bits=bits)
            for owner, bits in expected_activations.items()
            if owner not in rotation_owners}
        rotation = {
            owner: SimpleNamespace(bits=expected_activations[owner])
            for owner in rotation_owners}
        stem_contract = {
            "config": "STEM_W%dA%d" % (
                stem_weight_bits, stem_activation_bits),
            "weight_bits": stem_weight_bits,
            "activation_bits": stem_activation_bits,
        }

        runner.validate_configured_precision(
            current, expected_weights, ordinary, rotation, stem_contract)

        missing_weights = dict(expected_weights)
        del missing_weights[next(iter(missing_weights))]
        with self.assertRaisesRegex(RuntimeError, "weight bits differ"):
            runner.validate_configured_precision(
                current, missing_weights, ordinary, rotation, stem_contract)
        wrong_ordinary = dict(ordinary)
        first = next(iter(wrong_ordinary))
        wrong_ordinary[first] = SimpleNamespace(bits=8)
        if expected_activations[first] == 8:
            wrong_ordinary[first] = SimpleNamespace(bits=2)
        with self.assertRaisesRegex(RuntimeError, "activation bits differ"):
            runner.validate_configured_precision(
                current, expected_weights, wrong_ordinary, rotation,
                stem_contract)
        wrong_stem = dict(stem_contract)
        wrong_stem["activation_bits"] = (
            8 if stem_activation_bits != 8 else 2)
        with self.assertRaisesRegex(RuntimeError, "stem precision"):
            runner.validate_configured_precision(
                current, expected_weights, ordinary, rotation, wrong_stem)


class CandidateStatusTest(unittest.TestCase):
    @staticmethod
    def metrics():
        return {
            "RMSE": 0.2,
            "MAE": 0.1,
            "ABS_REL": 0.05,
            "IRMSE": 0.3,
            "flat_RMSE": 0.15,
            "boundary_RMSE": 0.25,
            "nonfinite_ratio": 0.0,
            "nonpositive_ratio": 0.0,
            "coefficient_sum_max_error": 0.0,
            "contraction_violation_ratio": 0.0,
            "anchor_max_error": 0.0,
        }

    @staticmethod
    def audit(feasible):
        return SimpleNamespace(
            feasible=feasible,
            weight_feasible=feasible,
            activation_feasible=feasible,
        )

    def test_validity_rejects_each_numeric_failure(self):
        failures = (
            ("RMSE", float("nan"), "nonfinite metric"),
            ("nonfinite_ratio", 0.1, "nonfinite prediction"),
            ("nonpositive_ratio", 0.1, "nonpositive depth"),
            ("coefficient_sum_max_error", 0.01, "coefficient sum"),
            ("contraction_violation_ratio", 0.01, "contraction"),
            ("anchor_max_error", 0.01, "anchor"),
        )
        for field, value, reason in failures:
            with self.subTest(field=field):
                metrics = self.metrics()
                metrics[field] = value
                status = runner.candidate_status(
                    metrics, self.audit(True), "joint")
                self.assertFalse(status.valid)
                self.assertIn(reason, status.reasons)

    def test_stage1_records_budget_excess_but_later_stages_reject_it(self):
        stage1 = runner.candidate_status(
            self.metrics(), self.audit(False), "single_block")
        joint = runner.candidate_status(
            self.metrics(), self.audit(False), "joint")

        self.assertTrue(stage1.valid)
        self.assertTrue(stage1.budget_excess)
        self.assertFalse(joint.valid)
        self.assertIn("precision budget", joint.reasons)


class CostBasisTest(unittest.TestCase):
    def test_cost_basis_requires_exact_registry_coverage(self):
        current = registry()
        weight_rows = tuple({
            "module": module,
            "macs": index + 1,
        } for index, module in enumerate(
            ordered_union(current.weights_by_block.values())))
        activation_rows = tuple({
            "module": owner[0],
            "kind": owner[1],
            "elements": index + 1,
        } for index, owner in enumerate(
            ordered_union(current.activations_by_block.values())))

        basis = runner.build_cost_basis(
            current, weight_rows, activation_rows)

        self.assertEqual(len(basis.weight_macs), 37)
        self.assertEqual(len(basis.activation_elements), 71)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            runner.build_cost_basis(
                current, weight_rows[:-1], activation_rows)


class FakeEvaluator(object):
    def __init__(self):
        self.phase_counts = []
        self.validation_names = ()

    @staticmethod
    def _row(candidate, phase, index):
        if candidate.assignment is None:
            score = 0.1
        else:
            score = 1.0 - 0.000001 * sum(
                bits for module, bits in candidate.assignment.weight_bits)
            score -= 0.000001 * sum(
                bits for owner, bits in candidate.assignment.activation_bits)
        if phase == "local":
            score += 1.0
        return {
            "config": candidate.name,
            "assignment": candidate.assignment,
            "calibration_RMSE": score + index * 1e-9,
            "boundary_RMSE": score + 0.1,
            "propagation_MSE": score + 0.2,
            "RMSE": score,
            "MAE": 0.1,
            "ABS_REL": 0.05,
            "IRMSE": 0.3,
            "flat_RMSE": 0.15,
            "nonfinite_ratio": 0.0,
            "nonpositive_ratio": 0.0,
            "coefficient_sum_max_error": 0.0,
            "contraction_violation_ratio": 0.0,
            "anchor_max_error": 0.0,
        }

    def calibration(self, phase, candidates):
        self.phase_counts.append((phase, len(candidates)))
        return tuple(
            self._row(candidate, phase, index)
            for index, candidate in enumerate(candidates))

    def validation(self, candidates):
        self.validation_names = tuple(
            candidate.name for candidate in candidates)
        return tuple(
            self._row(candidate, "validation", index)
            for index, candidate in enumerate(candidates))


class SearchOrchestrationTest(unittest.TestCase):
    @staticmethod
    def basis(current):
        return allocation.CostBasis(
            weight_macs=tuple(
                (module, 1) for module in ordered_union(
                    current.weights_by_block.values())),
            activation_elements=tuple(
                (owner, 1) for owner in ordered_union(
                    current.activations_by_block.values())),
        )

    def test_orchestration_runs_fixed_phases_and_freezes_before_validation(self):
        current = registry()
        evaluator = FakeEvaluator()
        protocol = runner.SearchProtocol(
            beam_width=512,
            joint_measured_limit=128,
            local_round_limit=3,
            refinement_block_limit=4,
            refinement_width=128,
            refinement_measured_limit=128,
        )

        result = runner.run_search(
            protocol, current, self.basis(current), evaluator)

        self.assertEqual(len(result.single_block_rows), 151)
        self.assertEqual(len(result.joint_rows), 128)
        self.assertLessEqual(len(result.local_rounds), 3)
        self.assertEqual(len(result.refined_rows), 128)
        self.assertEqual(
            evaluator.validation_names,
            ("FP32", "UNIFORM_W4A4", "CONTEXT_P3_T3_W8A8", "FINAL"))
        self.assertTrue(result.final_budget.feasible)
        self.assertEqual(
            result.final_assignment,
            next(candidate.assignment for candidate in result.validation_candidates
                 if candidate.name == "FINAL"))

    def test_validation_values_cannot_change_the_frozen_assignment(self):
        current = registry()
        basis = self.basis(current)
        protocol = runner.SearchProtocol(512, 128, 3, 4, 128, 128)
        left_evaluator = FakeEvaluator()
        right_evaluator = FakeEvaluator()

        left = runner.run_search(
            protocol, current, basis, left_evaluator)
        right = runner.run_search(
            protocol, current, basis, right_evaluator)

        self.assertEqual(left.final_assignment, right.final_assignment)


class AggregateCandidateTest(unittest.TestCase):
    def test_aggregate_uses_sample_means_and_worst_propagation_invariants(self):
        current = candidate()
        sample_rows = tuple({
            "sample_index": index,
            "RMSE": 0.2 + index * 0.1,
            "MAE": 0.1,
            "ABS_REL": 0.05,
            "IRMSE": 0.3,
            "flat_RMSE": 0.15,
            "boundary_RMSE": 0.25,
            "nonfinite_ratio": 0.0,
            "nonpositive_ratio": 0.0,
        } for index in range(2))
        propagation_rows = (
            {
                "signal": "affinity_constraints",
                "coefficient_sum_max_error": 0.0,
                "contraction_violation_rate": 0.0,
            },
            {
                "signal": "affinity_constraints",
                "coefficient_sum_max_error": 0.01,
                "contraction_violation_rate": 0.02,
            },
            {
                "signal": "anchor",
                "anchor_max_error": 0.03,
            },
        )

        row = runner.aggregate_candidate_result(
            current, sample_rows, propagation_rows, 2)

        self.assertAlmostEqual(row["calibration_RMSE"], 0.25)
        self.assertAlmostEqual(row["RMSE"], 0.25)
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["coefficient_sum_max_error"], 0.01)
        self.assertEqual(row["contraction_violation_ratio"], 0.02)
        self.assertEqual(row["anchor_max_error"], 0.03)
        self.assertEqual(row["assignment"], current.assignment)

    def test_aggregate_rejects_sample_identity_and_count_errors(self):
        current = candidate()
        rows = ({
            "sample_index": 1,
            "RMSE": 0.2,
            "MAE": 0.1,
            "ABS_REL": 0.05,
            "IRMSE": 0.3,
            "flat_RMSE": 0.15,
            "boundary_RMSE": 0.25,
            "nonfinite_ratio": 0.0,
            "nonpositive_ratio": 0.0,
        },)

        with self.assertRaisesRegex(ValueError, "sample count"):
            runner.aggregate_candidate_result(current, rows, (), 2)


if __name__ == "__main__":
    unittest.main()
