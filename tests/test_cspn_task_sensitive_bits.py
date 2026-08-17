import unittest

from spn_quant import cspn_encoder_prefix as encoder
from spn_quant import cspn_sensitivity as decoder
from spn_quant import cspn_task_sensitive_bits as allocation


def ordered_union(sequences):
    values = []
    for sequence in sequences:
        for value in sequence:
            if value not in values:
                values.append(value)
    return tuple(values)


def expected_modules():
    encoder_weights = (
        encoder.WEIGHT_MODULES_BY_UNIT["stem"] +
        encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer1"] +
        encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer2"] +
        encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer3"] +
        encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer4"])
    decoder_weights = (
        ("conv2",) +
        decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer1"] +
        decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer2"] +
        decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer3"] +
        decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer4"] +
        decoder.WEIGHT_MODULES_BY_BLOCK["initial_depth"])
    return encoder_weights + decoder_weights


def expected_owners():
    return ordered_union((
        encoder.ACTIVATION_OWNERS_BY_UNIT["stem"],
        encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer1"],
        encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer2"],
        encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer3"],
        encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer4"],
        decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer1"],
        decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer2"],
        decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer3"],
        decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer4"],
        decoder.ACTIVATION_OWNERS_BY_BLOCK["initial_depth"],
    ))


def registry():
    return allocation.build_registry(expected_modules(), expected_owners())


def unit_basis(current):
    modules = ordered_union(current.weights_by_block.values())
    owners = ordered_union(current.activations_by_block.values())
    return allocation.CostBasis(
        weight_macs=tuple((module, 1) for module in modules),
        activation_elements=tuple((owner, 1) for owner in owners),
    )


def probe_rows(current, validation_rmse):
    rows = []
    for probe in allocation.build_single_block_probes(current):
        block_order = (
            -1 if probe.block == "all"
            else allocation.BLOCK_ORDER.index(probe.block))
        rows.append({
            "config": probe.name,
            "calibration_RMSE": (
                1.0 + 0.01 * (8 - probe.weight_bits - probe.activation_bits)
                + 0.0001 * block_order),
            "boundary_RMSE": (
                0.5 + 0.005 *
                (8 - probe.weight_bits - probe.activation_bits)
                + 0.0001 * block_order),
            "propagation_MSE": (
                0.25 + 0.002 *
                (8 - probe.weight_bits - probe.activation_bits)
                + 0.0001 * block_order),
            "validation_RMSE": validation_rmse,
            "valid": True,
            "sensitivity_valid": True,
        })
    return rows


class AllocationRegistryTest(unittest.TestCase):
    def test_registry_covers_all_ten_blocks_and_official_sites(self):
        current = registry()

        self.assertEqual(tuple(current.weights_by_block), allocation.BLOCK_ORDER)
        self.assertEqual(
            tuple(current.activations_by_block), allocation.BLOCK_ORDER)
        self.assertEqual(
            len(ordered_union(current.weights_by_block.values())), 37)
        self.assertEqual(
            len(ordered_union(current.activations_by_block.values())), 71)
        self.assertIn("conv2", current.weights_by_block["decoder_layer1"])
        self.assertNotIn(
            ("rotation.layer4_signed_skip", "boundary"),
            current.activations_by_block["stem"])
        self.assertIn(
            ("rotation.layer4_signed_skip", "boundary"),
            current.activations_by_block["decoder_layer4"])
        self.assertEqual(
            sum(owner == ("rotation.layer4_signed_skip", "boundary")
                for owner in ordered_union(
                    current.activations_by_block.values())),
            1)

    def test_registry_rejects_missing_extra_and_duplicate_sites(self):
        modules = expected_modules()
        owners = expected_owners()

        with self.assertRaisesRegex(ValueError, "weight registry mismatch"):
            allocation.build_registry(modules[:-1], owners)
        with self.assertRaisesRegex(ValueError, "weight registry mismatch"):
            allocation.build_registry(modules + ("extra",), owners)
        with self.assertRaisesRegex(ValueError, "weight registry.*duplicates"):
            allocation.build_registry(modules + (modules[0],), owners)
        with self.assertRaisesRegex(ValueError, "activation registry mismatch"):
            allocation.build_registry(modules, owners[:-1])
        with self.assertRaisesRegex(ValueError, "activation registry.*duplicates"):
            allocation.build_registry(modules, owners + (owners[0],))

    def test_single_block_probes_have_exact_151_coverage(self):
        probes = allocation.build_single_block_probes(registry())

        self.assertEqual(len(probes), 151)
        self.assertEqual(probes[0].name, "UNIFORM_W4A4")
        self.assertEqual(len({probe.name for probe in probes}), 151)
        for block in allocation.BLOCK_ORDER:
            selected = [probe for probe in probes if probe.block == block]
            self.assertEqual(len(selected), 15)
            self.assertEqual(
                {(probe.weight_bits, probe.activation_bits)
                 for probe in selected},
                {(weight_bits, activation_bits)
                 for weight_bits in allocation.BIT_OPTIONS
                 for activation_bits in allocation.BIT_OPTIONS
                 if (weight_bits, activation_bits) != (4, 4)})


class BudgetTest(unittest.TestCase):
    @staticmethod
    def basis():
        return allocation.CostBasis(
            weight_macs=(("large", 90), ("small", 10)),
            activation_elements=(
                (("large", "input"), 80),
                (("small", "input"), 20),
            ))

    def test_budget_uses_macs_and_elements_not_site_counts(self):
        assignment = allocation.BitAssignment(
            weight_bits=(("large", 2), ("small", 8)),
            activation_bits=(
                (("large", "input"), 2),
                (("small", "input"), 8),
            ))

        audit = allocation.audit_budget(assignment, self.basis())

        self.assertEqual(audit.weight_numerator, 260)
        self.assertEqual(audit.weight_denominator, 100)
        self.assertEqual(audit.activation_numerator, 320)
        self.assertEqual(audit.activation_denominator, 100)
        self.assertAlmostEqual(audit.average_weight_bits, 2.6)
        self.assertAlmostEqual(audit.average_activation_bits, 3.2)
        self.assertTrue(audit.feasible)

    def test_budgets_are_independent_and_allow_exact_four(self):
        exact = allocation.BitAssignment(
            weight_bits=(("large", 4), ("small", 4)),
            activation_bits=(
                (("large", "input"), 4),
                (("small", "input"), 4),
            ))
        weight_excess = allocation.BitAssignment(
            weight_bits=(("large", 6), ("small", 2)),
            activation_bits=exact.activation_bits)
        activation_excess = allocation.BitAssignment(
            weight_bits=exact.weight_bits,
            activation_bits=(
                (("large", "input"), 6),
                (("small", "input"), 2),
            ))

        self.assertTrue(allocation.audit_budget(exact, self.basis()).feasible)
        self.assertFalse(
            allocation.audit_budget(weight_excess, self.basis()).weight_feasible)
        self.assertTrue(
            allocation.audit_budget(weight_excess, self.basis()).activation_feasible)
        self.assertTrue(
            allocation.audit_budget(activation_excess, self.basis()).weight_feasible)
        self.assertFalse(
            allocation.audit_budget(
                activation_excess, self.basis()).activation_feasible)

    def test_budget_requires_exact_positive_unique_cost_coverage(self):
        assignment = allocation.BitAssignment(
            weight_bits=(("large", 4), ("small", 4)),
            activation_bits=(
                (("large", "input"), 4),
                (("small", "input"), 4),
            ))

        with self.assertRaisesRegex(ValueError, "weight cost.*duplicates"):
            allocation.CostBasis(
                weight_macs=(("large", 90), ("large", 10)),
                activation_elements=self.basis().activation_elements)
        with self.assertRaisesRegex(ValueError, "positive"):
            allocation.CostBasis(
                weight_macs=(("large", 0), ("small", 10)),
                activation_elements=self.basis().activation_elements)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            allocation.audit_budget(
                assignment,
                allocation.CostBasis(
                    weight_macs=(("large", 90),),
                    activation_elements=self.basis().activation_elements))

    def test_per_bit_fractions_sum_to_one(self):
        assignment = allocation.BitAssignment(
            weight_bits=(("large", 2), ("small", 8)),
            activation_bits=(
                (("large", "input"), 2),
                (("small", "input"), 8),
            ))

        audit = allocation.audit_budget(assignment, self.basis())

        self.assertAlmostEqual(
            sum(value for bits, value in audit.weight_mac_fractions), 1.0)
        self.assertAlmostEqual(
            sum(value for bits, value in audit.activation_element_fractions),
            1.0)
        self.assertEqual(
            tuple(bits for bits, value in audit.weight_mac_fractions),
            allocation.BIT_OPTIONS)


class SearchTest(unittest.TestCase):
    def test_sensitivity_requires_complete_unique_finite_probe_rows(self):
        current = registry()
        probes = allocation.build_single_block_probes(current)
        rows = probe_rows(current, 0.1)

        table = allocation.build_sensitivity_table(probes, rows)

        self.assertEqual(len(table), 151)
        self.assertEqual(table[0].name, "UNIFORM_W4A4")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            allocation.build_sensitivity_table(probes, rows[:-1])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            allocation.build_sensitivity_table(
                probes, rows + [rows[0]])
        invalid = list(rows)
        invalid[1] = dict(invalid[1])
        invalid[1]["calibration_RMSE"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            allocation.build_sensitivity_table(probes, invalid)

        rejected = list(rows)
        rejected[1] = dict(rejected[1])
        rejected[1]["sensitivity_valid"] = False
        rejected[1]["calibration_RMSE"] = float("inf")
        table = allocation.build_sensitivity_table(probes, rejected)
        self.assertFalse(table[1].valid)
        self.assertEqual(table[1].calibration_rmse, float("inf"))

    def test_beam_is_deterministic_budgeted_and_ignores_validation(self):
        current = registry()
        basis = unit_basis(current)
        left_rows = probe_rows(current, 0.1)
        right_rows = list(reversed(probe_rows(current, 9.0)))

        left = allocation.search_block_assignments(
            current, basis, left_rows, 8, 4)
        right = allocation.search_block_assignments(
            current, basis, right_rows, 8, 4)

        self.assertEqual(left, right)
        self.assertEqual(len(left), 4)
        self.assertEqual(
            left, tuple(sorted(left, key=allocation.search_state_key)))
        self.assertTrue(all(
            allocation.audit_budget(state.assignment, basis).feasible
            for state in left))

    def test_dominance_pruning_removes_strictly_worse_state(self):
        best = allocation.SearchState(
            block_bits=(("stem", 2, 2),),
            estimated_rmse=1.0,
            estimated_boundary_rmse=0.5,
            estimated_propagation_mse=0.2,
            weight_numerator=10,
            activation_numerator=10,
            assignment=None,
        )
        worse = allocation.SearchState(
            block_bits=(("stem", 4, 4),),
            estimated_rmse=1.1,
            estimated_boundary_rmse=0.6,
            estimated_propagation_mse=0.3,
            weight_numerator=20,
            activation_numerator=20,
            assignment=None,
        )

        self.assertEqual(
            allocation.prune_dominated_states((worse, best)), (best,))


class LocalAndRefinementTest(unittest.TestCase):
    def test_neighbors_are_canonical_complete_and_budget_preserving(self):
        current_registry = registry()
        basis = unit_basis(current_registry)
        current = allocation.uniform_assignment(current_registry, 4, 4)

        neighbors = allocation.build_budget_preserving_neighbors(
            current, current_registry, basis)

        self.assertTrue(neighbors)
        self.assertEqual(
            neighbors, tuple(sorted(neighbors, key=allocation.assignment_key)))
        self.assertEqual(len(neighbors), len(set(neighbors)))
        for neighbor in neighbors:
            self.assertTrue(allocation.audit_budget(neighbor, basis).feasible)
            changed = [
                abs(dict(current.weight_bits)[module] - bits)
                for module, bits in neighbor.weight_bits
                if dict(current.weight_bits)[module] != bits]
            changed.extend(
                abs(dict(current.activation_bits)[owner] - bits)
                for owner, bits in neighbor.activation_bits
                if dict(current.activation_bits)[owner] != bits)
            self.assertTrue(changed)
            self.assertEqual(set(changed), {2})

    def test_local_selection_requires_complete_rows_and_strict_improvement(self):
        current_registry = registry()
        basis = unit_basis(current_registry)
        current = allocation.uniform_assignment(current_registry, 4, 4)
        neighbors = allocation.build_budget_preserving_neighbors(
            current, current_registry, basis)[:3]
        rows = tuple({
            "assignment": neighbor,
            "calibration_RMSE": 0.9 + 0.01 * index,
            "boundary_RMSE": 0.4,
            "propagation_MSE": 0.2,
        } for index, neighbor in enumerate(neighbors))

        selected = allocation.select_local_improvement(
            current, 1.0, neighbors, rows, basis)

        self.assertEqual(selected, neighbors[0])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            allocation.select_local_improvement(
                current, 1.0, neighbors, rows[:-1], basis)
        no_improvement = tuple(
            dict(row, calibration_RMSE=1.0) for row in rows)
        self.assertEqual(
            allocation.select_local_improvement(
                current, 1.0, neighbors, no_improvement, basis),
            current)

    def test_refinement_ranking_uses_measured_cheapest_demotions(self):
        current_registry = registry()
        basis = unit_basis(current_registry)
        current = allocation.uniform_assignment(current_registry, 4, 4)
        demotions = allocation.build_cheapest_block_demotions(
            current, current_registry, basis)
        rows = tuple({
            "block": block,
            "assignment": demotions[block],
            "calibration_RMSE": 1.0 + 0.01 * index,
            "boundary_RMSE": 0.5,
            "propagation_MSE": 0.25,
        } for index, block in enumerate(allocation.BLOCK_ORDER))

        selected = allocation.rank_refinement_blocks(
            current, current_registry, basis, 1.0, rows, 4)

        self.assertEqual(
            selected, tuple(reversed(allocation.BLOCK_ORDER[-4:])))
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            allocation.rank_refinement_blocks(
                current, current_registry, basis, 1.0, rows[:-1], 4)

    def test_refinement_beam_changes_only_selected_sites(self):
        current_registry = registry()
        basis = unit_basis(current_registry)
        current = allocation.uniform_assignment(current_registry, 4, 4)
        selected = ("stem", "encoder_layer1")

        left = allocation.build_refinement_candidates(
            current, current_registry, basis, selected,
            probe_rows(current_registry, 0.1), 32, 8)
        right = allocation.build_refinement_candidates(
            current, current_registry, basis, selected,
            list(reversed(probe_rows(current_registry, 9.0))), 32, 8)

        self.assertEqual(left, right)
        self.assertEqual(len(left), 8)
        selected_weights = set(ordered_union(
            current_registry.weights_by_block[block] for block in selected))
        selected_owners = set(ordered_union(
            current_registry.activations_by_block[block]
            for block in selected))
        current_weights = dict(current.weight_bits)
        current_activations = dict(current.activation_bits)
        for state in left:
            self.assertIsNotNone(state.assignment)
            self.assertTrue(
                allocation.audit_budget(state.assignment, basis).feasible)
            self.assertTrue(all(
                module in selected_weights or bits == current_weights[module]
                for module, bits in state.assignment.weight_bits))
            self.assertTrue(all(
                owner in selected_owners or bits == current_activations[owner]
                for owner, bits in state.assignment.activation_bits))


if __name__ == "__main__":
    unittest.main()
