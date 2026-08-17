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


if __name__ == "__main__":
    unittest.main()
