import pytest

from spn_quant import cspn_task_sensitive_bits as cspn_allocation
from spn_quant import mixed_precision
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


def model_contract():
    blocks = tuple(
        QuantizationBlock(
            name,
            ("weight_%s" % name,),
            (("edge_%s" % name, "input"),),
        )
        for name in ("alpha", "beta", "gamma", "delta")
    )
    return QuantizationModelContract(
        model_name="neutral_model",
        blocks=blocks,
        prefix_groups=(
            ("alpha",),
            ("alpha", "beta"),
            ("alpha", "beta", "gamma"),
        ),
        tail_groups=(("gamma",), ("delta",)),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("protected.projection",),
        module_roles=(("protected.projection", "propagation_state"),),
    )


def cost_basis():
    return mixed_precision.CostBasis(
        weight_macs=(
            ("weight_alpha", 1),
            ("weight_beta", 2),
            ("weight_gamma", 4),
            ("weight_delta", 3),
        ),
        activation_elements=(
            (("edge_alpha", "input"), 2),
            (("edge_beta", "input"), 2),
            (("edge_gamma", "input"), 3),
            (("edge_delta", "input"), 3),
        ),
    )


def probe_rows(registry):
    rows = []
    for index, probe in enumerate(
            mixed_precision.build_single_block_probes(registry)):
        rows.append({
            "config": probe.name,
            "calibration_RMSE": 1.0 + index * 0.0001,
            "boundary_RMSE": 0.5 + index * 0.0001,
            "propagation_MSE": 0.25 + index * 0.0001,
            "sensitivity_valid": True,
        })
    return tuple(rows)


def test_registry_uses_contract_blocks_without_cspn_names():
    contract = model_contract()

    registry = mixed_precision.build_registry(contract, cost_basis())

    assert registry.model_name == "neutral_model"
    assert registry.blocks == ("alpha", "beta", "gamma", "delta")
    assert tuple(registry.weights_by_block) == registry.blocks
    assert tuple(registry.activations_by_block) == registry.blocks
    assert "stem" not in registry.weights_by_block


def test_registry_rejects_cost_rows_outside_contract():
    basis = cost_basis()
    changed = mixed_precision.CostBasis(
        weight_macs=basis.weight_macs[:-1] + (("weight_extra", 3),),
        activation_elements=basis.activation_elements,
    )

    with pytest.raises(ValueError, match="weight cost coverage mismatch"):
        mixed_precision.build_registry(model_contract(), changed)


def test_generic_registry_does_not_infer_missing_contract_identity():
    mappings = {"alpha": ("weight_alpha",)}
    owners = {"alpha": (("edge_alpha", "input"),)}

    with pytest.raises(TypeError):
        mixed_precision.AllocationRegistry(mappings, owners)

    legacy = cspn_allocation.AllocationRegistry(mappings, owners)
    assert legacy.weights_by_block == mappings
    assert legacy.activations_by_block == owners


def test_generic_assignment_and_search_use_registry_model_and_block_order():
    registry = mixed_precision.build_registry(model_contract(), cost_basis())
    baseline = mixed_precision.uniform_assignment(registry, 4, 4)

    assert baseline.model_name == "neutral_model"
    assert tuple(module for module, bits in baseline.weight_bits) == (
        "weight_alpha", "weight_beta", "weight_delta", "weight_gamma")
    probes = mixed_precision.build_single_block_probes(registry)
    assert len(probes) == 61
    assert {probe.block for probe in probes[1:]} == set(registry.blocks)

    states = mixed_precision.search_block_assignments(
        registry, cost_basis(), probe_rows(registry), 16, 4)

    assert len(states) == 4
    assert all(state.assignment.model_name == "neutral_model"
               for state in states)
    assert all(tuple(block for block, weight_bits, activation_bits
                     in state.block_bits) == registry.blocks
               for state in states)


def test_cspn_public_records_are_compatibility_aliases_with_same_tuple_payload():
    assignment = cspn_allocation.BitAssignment(
        (("module", 4),),
        ((("owner", "input"), 8),),
    )

    assert cspn_allocation.BitAssignment is mixed_precision.BitAssignment
    assert cspn_allocation.CostBasis is mixed_precision.CostBasis
    assert cspn_allocation.SearchState is mixed_precision.SearchState
    assert assignment.model_name == ""
    assert assignment.weight_bits == (("module", 4),)
    assert assignment.activation_bits == ((("owner", "input"), 8),)
