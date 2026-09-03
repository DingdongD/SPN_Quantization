import torch
import pytest

from scripts import prepare_nyu_three_model_static_inputs as prepare
from spn_quant.nyu_static_inputs import ACTIVATION_DESCRIPTOR_FIELDS


def test_single_channel_initial_depth_imbalance_is_diagnostic():
    for model in ("dyspn", "nlspn", "completionformer"):
        diagnostics = prepare.activation_descriptor_diagnostic(model)
        fields = ACTIVATION_DESCRIPTOR_FIELDS[model]
        assert len(diagnostics) == len(fields)
        assert {
            name for name, diagnostic in zip(fields, diagnostics)
            if diagnostic
        } == {"%s_initial_depth_channel_imbalance" % model}


def test_concat_cost_hook_counts_one_declared_branch_without_offset_error():
    capture = object.__new__(prepare.CostCapture)
    owner = ("concat::fusion::transformer_input", "concat_branch")
    capture.activation_elements = {owner: 0}
    tensor = torch.ones(1, 8, 4, 4)

    capture._concat_input_hook(owner, 0)(None, (tensor,))

    assert capture.activation_elements[owner] == tensor.numel() // 2


def test_static_output_directory_preserves_only_artifacts_subdirectory(tmp_path):
    output_root = tmp_path / "model"
    output_root.mkdir()
    (output_root / "artifacts").mkdir()
    destinations = (output_root / "calibration_metadata.json",)

    prepare.validate_static_output_directory(output_root, destinations)

    (output_root / "calibration_metadata.json").touch()
    with pytest.raises(FileExistsError, match="static-input output"):
        prepare.validate_static_output_directory(output_root, destinations)
