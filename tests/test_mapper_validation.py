import pytest

from spn_quant.hardware.mapper_validation import (
    BackendResult,
    deduplicate_gemms,
    dense_gemm_instances,
    validate_backend_result,
)
from spn_quant.hardware.workload import LayerWorkload, WorkloadProfile


def _profile(layers):
    return WorkloadProfile(
        model_name="cspn",
        source_config="source.json",
        checkpoint="/cspn.pt",
        sample_index=0,
        input_shapes=(("arg0", (1, 4, 8, 8)),),
        layers=tuple(layers),
    )


def test_dense_instances_exclude_propagation():
    profile = _profile((
        LayerWorkload.linear(
            "dense", (1, 8), (1, 4), execution_index=0),
        LayerWorkload.irregular(
            "prop", "propagation", (1, 4, 8, 8), (1, 4, 8, 8),
            macs=7, execution_index=1),
    ))

    rows = dense_gemm_instances(profile)

    assert [(row.layer_name, row.kind) for row in rows] == [
        ("dense", "linear")]
    assert rows[0].workload.macs == profile.layers[0].macs


def test_unique_gemms_preserve_all_original_instances():
    profile = _profile((
        LayerWorkload.linear(
            "dense_a", (1, 8), (1, 4), execution_index=0),
        LayerWorkload.linear(
            "dense_b", (1, 8), (1, 4), execution_index=1),
    ))

    unique = deduplicate_gemms(dense_gemm_instances(profile))

    assert len(unique) == 1
    assert unique[0].gemm_id == "gemm_0000"
    assert [row.layer_name for row in unique[0].instances] == [
        "dense_a", "dense_b"]


def test_backend_result_rejects_wrong_mac_count():
    unique, = deduplicate_gemms(dense_gemm_instances(_profile((
        LayerWorkload.linear("dense", (1, 8), (1, 4)),
    ))))
    result = BackendResult(
        backend="factorflow", architecture_id="balanced__32x32",
        gemm_id=unique.gemm_id, macs=unique.workload.macs - 1,
        cycles=1, energy_uj=1.0, utilization=1.0,
        search_seconds=0.1, native_objective="edp")

    with pytest.raises(ValueError, match="MAC count"):
        validate_backend_result(unique, result)


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.0001])
def test_backend_result_rejects_nonphysical_utilization(utilization):
    with pytest.raises(ValueError, match="utilization"):
        BackendResult(
            backend="tcm", architecture_id="balanced__32x32",
            gemm_id="gemm_0000", macs=32, cycles=1,
            energy_uj=1.0, utilization=utilization,
            search_seconds=0.1, native_objective="edp")
