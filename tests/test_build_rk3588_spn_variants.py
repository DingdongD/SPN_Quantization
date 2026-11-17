from pathlib import Path

import onnx
from onnx import TensorProto, helper

from scripts import build_rk3588_spn_variants as builder


def _identity_graph(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["rgb"], ["pred"], name="identity")],
        "tiny",
        [helper.make_tensor_value_info(
            "rgb", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info(
            "pred", TensorProto.FLOAT, [1, 1, 2, 2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_variant_plan_has_exact_ab_candidates():
    assert tuple(item.name for item in builder.variant_plan()) == (
        "compressed_fp16_legacy_grid",
        "compressed_fp16_packed_grid",
    )
    assert tuple(item.pack_grid_samples for item in builder.variant_plan()) == (
        False, True)


def test_graph_audit_records_interface_operators_and_hash(tmp_path: Path):
    graph = tmp_path / "tiny.onnx"
    _identity_graph(graph)

    audit = builder.audit_graph(graph)

    assert audit["sha256"]
    assert audit["inputs"] == ["rgb"]
    assert audit["outputs"] == ["pred"]
    assert audit["operator_counts"] == {"Identity": 1}
    assert audit["deformable_count"] == 0
    assert audit["grid_sample_count"] == 0
    assert audit["conv_transpose_count"] == 0


def test_validate_variant_pair_requires_packed_reduction():
    legacy = {
        "inputs": ["rgb", "dep"], "outputs": ["pred"],
        "deformable_count": 0, "grid_sample_count": 30,
    }
    packed = dict(legacy, grid_sample_count=6)

    builder.validate_variant_pair(legacy, packed)

    packed["grid_sample_count"] = 30
    try:
        builder.validate_variant_pair(legacy, packed)
    except ValueError as error:
        assert "reduce GridSample" in str(error)
    else:
        raise AssertionError("unchanged GridSample count was accepted")
