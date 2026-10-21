import json

import pytest

from spn_quant.hardware.workload import (
    LayerWorkload,
    WorkloadProfile,
    conv2d_to_gemms,
    conv_transpose2d_to_gemms,
    linear_to_gemms,
)


def test_conv2d_group_lowering_preserves_macs():
    layer = LayerWorkload.conv2d(
        "group", [1, 32, 28, 38], [1, 64, 28, 38],
        kernel=(3, 3), stride=(1, 1), groups=4)
    gemms = conv2d_to_gemms(layer)
    assert [(g.m, g.n, g.k, g.repetitions) for g in gemms] == [
        (28 * 38, 16, 8 * 3 * 3, 4)]
    assert sum(g.macs for g in gemms) == layer.macs


def test_transposed_conv_uses_input_stationary_gemm_shape():
    layer = LayerWorkload.conv_transpose2d(
        "up", [1, 64, 14, 19], [1, 32, 28, 38],
        kernel=(4, 4), stride=(2, 2), groups=1)
    gemm, = conv_transpose2d_to_gemms(layer)
    assert (gemm.m, gemm.n, gemm.k) == (14 * 19, 32 * 4 * 4, 64)
    assert gemm.macs == layer.macs


def test_linear_flattens_all_leading_dimensions():
    layer = LayerWorkload.linear("fc", [2, 7, 64], [2, 7, 96])
    gemm, = linear_to_gemms(layer)
    assert (gemm.m, gemm.n, gemm.k) == (14, 96, 64)
    assert gemm.macs == layer.macs


def test_profile_json_round_trip_is_stable(tmp_path):
    layer = LayerWorkload.linear(
        "fc", [1, 64], [1, 8], execution_index=3)
    profile = WorkloadProfile(
        model_name="cspn",
        source_config="configs/source.json",
        checkpoint="/checkpoints/cspn.pt",
        sample_index=386,
        input_shapes=(("rgb", (1, 3, 228, 304)),),
        layers=(layer,),
        metadata=(("torch_version", "2.7.1"),),
    )
    path = tmp_path / "profile.json"
    profile.write_json(path)
    assert WorkloadProfile.read_json(path) == profile
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 1


def test_layer_rejects_zero_dimension():
    with pytest.raises(ValueError, match="positive"):
        LayerWorkload.linear("bad", [1, 0], [1, 8])
