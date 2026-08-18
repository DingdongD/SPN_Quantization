import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

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


class FixedSampleTest(unittest.TestCase):
    def test_seeded_sample_is_atomic_across_threads(self):
        class RandomDataset(object):
            def __getitem__(self, index):
                time.sleep(0.01)
                return (
                    float(np.random.uniform()),
                    float(torch.rand(1).item()),
                )

        dataset = RandomDataset()
        indices = tuple(range(8))
        expected = tuple(
            runner.seeded_sample(dataset, index, 100)
            for index in indices)

        with ThreadPoolExecutor(max_workers=4) as executor:
            observed = tuple(executor.map(
                lambda index: runner.seeded_sample(dataset, index, 100),
                indices))

        self.assertEqual(observed, expected)

    def test_materialized_samples_preserve_declared_indices(self):
        class SourceDataset(object):
            def __init__(self):
                self.calls = []

            def __len__(self):
                return 10

            def __getitem__(self, index):
                self.calls.append(index)
                return {"value": torch.tensor([index])}

        source = SourceDataset()

        fixed = runner.materialize_samples(source, (7, 2), 100)

        self.assertEqual(len(fixed), len(source))
        self.assertEqual(source.calls, [7, 2])
        self.assertEqual(int(fixed[7]["value"].item()), 7)
        self.assertEqual(int(fixed[2]["value"].item()), 2)
        with self.assertRaises(KeyError):
            fixed[0]


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
            "valid": True,
            "sensitivity_valid": True,
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

    def test_nonpositive_depth_is_reported_without_rejecting_search(self):
        metrics = self.metrics()
        metrics["nonpositive_ratio"] = 0.1

        status = runner.candidate_status(
            metrics, self.audit(True), "joint")

        self.assertTrue(status.valid)
        self.assertEqual(status.reasons, ())

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
            "sensitivity_valid": True,
            "valid": True,
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

    def test_refinement_cannot_replace_a_better_incumbent(self):
        class WorseRefinementEvaluator(FakeEvaluator):
            def calibration(self, phase, candidates):
                rows = super().calibration(phase, candidates)
                if phase != "refinement":
                    return rows
                return tuple(dict(
                    row,
                    calibration_RMSE=float(row["calibration_RMSE"]) + 10.0,
                    RMSE=float(row["RMSE"]) + 10.0,
                ) for row in rows)

        current = registry()
        evaluator = WorseRefinementEvaluator()
        protocol = runner.SearchProtocol(512, 128, 3, 4, 128, 128)

        result = runner.run_search(
            protocol, current, self.basis(current), evaluator)
        incumbent = min(
            result.joint_rows,
            key=lambda row: float(row["calibration_RMSE"]))

        self.assertEqual(result.final_assignment, incumbent["assignment"])


class AggregateCandidateTest(unittest.TestCase):
    def test_aggregate_uses_pixel_weighted_regions_when_a_sample_has_no_boundary(self):
        current = candidate()
        sample_rows = (
            {
                "sample_index": 0,
                "RMSE": 0.2,
                "MAE": 0.1,
                "ABS_REL": 0.05,
                "IRMSE": 0.3,
                "flat_RMSE": 0.15,
                "boundary_RMSE": float("nan"),
                "nonfinite_ratio": 0.0,
                "nonpositive_ratio": 0.0,
            },
            {
                "sample_index": 1,
                "RMSE": 0.4,
                "MAE": 0.2,
                "ABS_REL": 0.1,
                "IRMSE": 0.6,
                "flat_RMSE": 0.3,
                "boundary_RMSE": 0.5,
                "nonfinite_ratio": 0.0,
                "nonpositive_ratio": 0.0,
            },
        )
        region_rows = (
            {"region": "smooth", "num_pixels": 3, "sum_sq": 0.12,
             "sum_abs": 0.5, "sum_abs_rel": 0.2},
            {"region": "smooth", "num_pixels": 1, "sum_sq": 0.04,
             "sum_abs": 0.2, "sum_abs_rel": 0.1},
            {"region": "boundary", "num_pixels": 0, "sum_sq": 0.0,
             "sum_abs": 0.0, "sum_abs_rel": 0.0},
            {"region": "boundary", "num_pixels": 4, "sum_sq": 1.0,
             "sum_abs": 2.0, "sum_abs_rel": 0.8},
        )
        propagation_rows = (
            {"signal": "state", "mse": 0.04},
            {"signal": "affinity_constraints",
             "coefficient_sum_max_error": 0.0,
             "contraction_violation_rate": 0.0},
            {"signal": "anchor", "anchor_max_error": 0.0},
        )

        row = runner.aggregate_candidate_result(
            current, sample_rows, region_rows, propagation_rows, 2)

        self.assertAlmostEqual(row["flat_RMSE"], 0.2)
        self.assertAlmostEqual(row["boundary_RMSE"], 0.5)
        self.assertTrue(row["sensitivity_valid"])

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
                "signal": "state",
                "mse": 0.04,
            },
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
        region_rows = (
            {"region": "smooth", "num_pixels": 2, "sum_sq": 0.045,
             "sum_abs": 0.3, "sum_abs_rel": 0.1},
            {"region": "boundary", "num_pixels": 2, "sum_sq": 0.125,
             "sum_abs": 0.5, "sum_abs_rel": 0.2},
        )

        row = runner.aggregate_candidate_result(
            current, sample_rows, region_rows, propagation_rows, 2)

        self.assertAlmostEqual(row["calibration_RMSE"], 0.25)
        self.assertAlmostEqual(row["RMSE"], 0.25)
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["coefficient_sum_max_error"], 0.01)
        self.assertEqual(row["contraction_violation_ratio"], 0.02)
        self.assertEqual(row["anchor_max_error"], 0.03)
        self.assertEqual(row["propagation_MSE"], 0.04)
        self.assertFalse(row["valid"])
        self.assertTrue(row["sensitivity_valid"])
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
            runner.aggregate_candidate_result(current, rows, (), (), 2)

    def test_aggregate_retains_nonfinite_candidate_as_invalid(self):
        current = candidate()
        rows = ({
            "sample_index": 1,
            "RMSE": float("inf"),
            "MAE": 0.1,
            "ABS_REL": 0.05,
            "IRMSE": 0.3,
            "flat_RMSE": 0.15,
            "boundary_RMSE": 0.25,
            "nonfinite_ratio": 0.0,
            "nonpositive_ratio": 0.0,
        },)
        propagation = (
            {"signal": "state", "mse": 0.1},
            {"signal": "affinity_constraints",
             "coefficient_sum_max_error": 0.0,
             "contraction_violation_rate": 0.0},
            {"signal": "anchor", "anchor_max_error": 0.0},
        )
        region_rows = (
            {"region": "smooth", "num_pixels": 1, "sum_sq": 0.0225,
             "sum_abs": 0.15, "sum_abs_rel": 0.05},
            {"region": "boundary", "num_pixels": 1, "sum_sq": 0.0625,
             "sum_abs": 0.25, "sum_abs_rel": 0.1},
        )

        row = runner.aggregate_candidate_result(
            current, rows, region_rows, propagation, 1)

        self.assertFalse(row["valid"])
        self.assertFalse(row["sensitivity_valid"])
        self.assertEqual(row["RMSE"], float("inf"))


class FakeWorker(object):
    def __init__(self, worker_id, basis):
        self.worker_id = worker_id
        self.basis = basis
        self.calibration_calls = []
        self.validation_calls = []
        self.closed = False

    def calibration(self, phase, candidates):
        self.calibration_calls.append(
            (phase, tuple(candidate.name for candidate in candidates)))
        return tuple({
            "config": candidate.name,
            "assignment": candidate.assignment,
            "worker": self.worker_id,
        } for candidate in candidates)

    def validation(self, candidates):
        self.validation_calls.append(
            tuple(candidate.name for candidate in candidates))
        return tuple({
            "config": candidate.name,
            "assignment": candidate.assignment,
            "worker": self.worker_id,
        } for candidate in candidates)

    def cost_basis(self):
        return self.basis

    def close(self):
        self.closed = True


class ParallelEvaluatorTest(unittest.TestCase):
    def test_coordinator_device_is_first_declared_cuda_device(self):
        self.assertEqual(
            runner.coordinator_device(("cuda:2", "cuda:3")), "cuda:2")

    def test_parallel_evaluator_uses_stable_round_robin_and_input_order(self):
        current = registry()
        basis = SearchOrchestrationTest.basis(current)
        workers = tuple(FakeWorker(index, basis) for index in range(3))
        evaluator = runner.ParallelEvaluator(workers)
        assignments = allocation.uniform_assignment(current, 4, 4)
        candidates = tuple(
            runner.RuntimeCandidate("C%d" % index, "joint", assignments)
            for index in range(8))

        rows = evaluator.calibration("joint", candidates)

        self.assertEqual(
            tuple(row["config"] for row in rows),
            tuple(candidate.name for candidate in candidates))
        self.assertEqual(
            workers[0].calibration_calls,
            [("joint", ("C0",)), ("joint", ("C3",)),
             ("joint", ("C6",))])
        self.assertEqual(
            workers[1].calibration_calls,
            [("joint", ("C1",)), ("joint", ("C4",)),
             ("joint", ("C7",))])
        self.assertEqual(
            workers[2].calibration_calls,
            [("joint", ("C2",)), ("joint", ("C5",))])
        self.assertEqual(evaluator.cost_basis(), basis)
        evaluator.close()
        self.assertTrue(all(worker.closed for worker in workers))

    def test_parallel_evaluator_rejects_worker_cost_drift(self):
        current = registry()
        basis = SearchOrchestrationTest.basis(current)
        changed = allocation.CostBasis(
            weight_macs=tuple(
                (module, cost + (1 if index == 0 else 0))
                for index, (module, cost) in enumerate(basis.weight_macs)),
            activation_elements=basis.activation_elements,
        )
        evaluator = runner.ParallelEvaluator((
            FakeWorker(0, basis), FakeWorker(1, changed)))
        assignment = allocation.uniform_assignment(current, 4, 4)
        candidates = (
            runner.RuntimeCandidate("LEFT", "joint", assignment),
            runner.RuntimeCandidate("RIGHT", "joint", assignment),
        )

        with self.assertRaisesRegex(RuntimeError, "cost basis differs"):
            evaluator.calibration("joint", candidates)

    def test_explicit_phase_cache_resumes_completed_candidates(self):
        current = registry()
        basis = SearchOrchestrationTest.basis(current)
        assignment = allocation.uniform_assignment(current, 4, 4)
        candidates = tuple(
            runner.RuntimeCandidate("CACHE_%d" % index, "joint", assignment)
            for index in range(3))
        root = Path("tests/.cspn_task_sensitive_phase_cache")
        if root.exists():
            shutil.rmtree(root)
        first_workers = tuple(FakeWorker(index, basis) for index in range(2))
        first = runner.ParallelEvaluator(first_workers, root)
        expected = first.calibration("joint", candidates)
        self.assertEqual(first.cost_basis(), basis)
        first.close()

        resumed_workers = tuple(FakeWorker(index, basis) for index in range(2))
        resumed = runner.ParallelEvaluator(resumed_workers, root)
        observed = resumed.calibration("joint", candidates)

        self.assertEqual(observed, expected)
        self.assertTrue(all(
            not worker.calibration_calls for worker in resumed_workers))
        self.assertEqual(resumed.cost_basis(), basis)
        resumed.close()
        shutil.rmtree(root)


class OutputContractTest(unittest.TestCase):
    def test_prediction_writer_receives_staging_output_root(self):
        root = Path("tests/.cspn_task_sensitive_prediction_root")
        staging = Path(str(root) + ".incomplete")
        if root.exists():
            shutil.rmtree(root)
        if staging.exists():
            shutil.rmtree(staging)
        staging = runner.prepare_output_directories(root, False)

        output_root = runner.prediction_output_root(staging)

        self.assertEqual(output_root, staging)
        self.assertEqual(
            runner.prepare_prediction_dir(output_root, "FP32"),
            staging / "predictions" / "FP32")
        shutil.rmtree(staging)

    def test_final_allocation_rows_have_exact_cost_and_fraction_coverage(self):
        current = registry()
        basis = SearchOrchestrationTest.basis(current)
        final = allocation.uniform_assignment(current, 4, 4)

        rows = runner.final_allocation_rows(current, basis, final)

        weight_rows = tuple(row for row in rows if row["tensor"] == "weight")
        activation_rows = tuple(
            row for row in rows if row["tensor"] == "activation")
        self.assertEqual(len(weight_rows), 37)
        self.assertEqual(len(activation_rows), 71)
        self.assertAlmostEqual(
            sum(float(row["cost_fraction"]) for row in weight_rows), 1.0)
        self.assertAlmostEqual(
            sum(float(row["cost_fraction"]) for row in activation_rows),
            1.0)
        skip = tuple(
            row for row in activation_rows
            if row["module"] == "rotation.layer4_signed_skip")
        self.assertEqual(len(skip), 1)
        self.assertEqual(skip[0]["block"], "decoder_layer4")

    def test_production_protocol_requires_the_approved_search_limits(self):
        approved = runner.SearchProtocol(512, 128, 3, 4, 128, 128)

        runner.validate_production_protocol(approved)

        with self.assertRaisesRegex(ValueError, "production protocol"):
            runner.validate_production_protocol(
                runner.SearchProtocol(256, 128, 3, 4, 128, 128))

    def test_persisted_fp32_row_has_explicit_precision_marker(self):
        row = runner._persisted_metric_row({
            "config": "FP32",
            "assignment": None,
            "RMSE": 0.1,
        }, "validation", 0)

        self.assertEqual(row["assignment"], "FP32")

    def test_publish_renames_complete_staging_tree_once(self):
        current = registry()
        basis = SearchOrchestrationTest.basis(current)
        final = allocation.uniform_assignment(current, 4, 4)
        audit = allocation.audit_budget(final, basis)
        root = Path("tests/.cspn_task_sensitive_publish_test")
        staging = Path(str(root) + ".incomplete")
        if root.exists():
            shutil.rmtree(root)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        (staging / "predictions").mkdir()
        metric = {
            "config": "CANDIDATE",
            "assignment": final,
            "calibration_RMSE": 0.2,
            "boundary_RMSE": 0.3,
            "propagation_MSE": 0.01,
            "RMSE": 0.2,
        }
        fp32 = dict(metric)
        fp32["config"] = "FP32"
        fp32["assignment"] = None
        result = runner.SearchResult(
            single_block_rows=(metric,),
            joint_rows=(metric,),
            local_rounds=(),
            demotion_rows=(dict(metric, block="stem"),),
            refined_rows=(metric,),
            validation_rows=(fp32, metric),
            validation_candidates=(
                runner.ValidationCandidate("FP32", None),
                runner.ValidationCandidate("CANDIDATE", final),
            ),
            refinement_blocks=("stem",),
            final_assignment=final,
            final_budget=audit,
        )

        runner.publish_search_result(
            staging, root, result, current, basis,
            runner.SearchProtocol(512, 128, 3, 4, 128, 128))

        self.assertTrue(root.is_dir())
        self.assertFalse(staging.exists())
        self.assertTrue((root / "manifest.json").is_file())
        self.assertTrue((root / "final_allocation.csv").is_file())
        shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
