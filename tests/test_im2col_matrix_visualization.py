from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spn_quant.im2col_matrix_visualization import (
    ConvMatrixCapture,
    FullMatrixCaptureRecorder,
    full_line_curtain,
)


class UniformQuantizer:
    format = "uniform"
    bits = 8
    unsigned = False
    zero_point = 0
    qmin = -127
    qmax = 127

    def __init__(self, scale):
        self.scale = float(scale)

    def scale_for(self, tensor):
        del tensor
        return self.scale


def _capture():
    layer = nn.Conv2d(
        2, 3, kernel_size=(2, 3), stride=(2, 1),
        padding=(1, 0), bias=False)
    inputs = torch.arange(1 * 2 * 5 * 6, dtype=torch.float32).reshape(
        1, 2, 5, 6) / 8.0
    quantized_inputs = torch.round(inputs / 0.125) * 0.125
    quantized_weight = torch.round(layer.weight.detach() / 0.01) * 0.01
    capture = ConvMatrixCapture.from_tensors(
        module="conv", sample_index=7, layer=layer,
        reference_input=inputs, quantized_input=quantized_inputs,
        original_weight=layer.weight.detach(),
        quantized_weight=quantized_weight,
        activation_bits=8, activation_unsigned=False,
        activation_scale=torch.tensor(0.125, dtype=torch.float32))
    return capture, layer, inputs, quantized_inputs, quantized_weight


def test_capture_reconstructs_complete_weight_and_activation_matrices():
    capture, layer, inputs, quantized_inputs, quantized_weight = _capture()

    matrices = capture.matrices()

    expected_reference = F.unfold(
        inputs, layer.kernel_size, dilation=layer.dilation,
        padding=layer.padding, stride=layer.stride)
    expected_quantized = F.unfold(
        quantized_inputs, layer.kernel_size, dilation=layer.dilation,
        padding=layer.padding, stride=layer.stride)
    assert matrices.reference_weight.shape == (3, 12)
    assert matrices.reference_activation.shape == (
        expected_reference.shape[2], 12)
    torch.testing.assert_close(
        matrices.reference_activation,
        expected_reference[0].transpose(0, 1))
    torch.testing.assert_close(
        matrices.quantized_activation,
        expected_quantized[0].transpose(0, 1))
    torch.testing.assert_close(
        matrices.reference_weight, layer.weight.detach().reshape(3, 12))
    torch.testing.assert_close(
        matrices.quantized_weight, quantized_weight.reshape(3, 12))
    torch.testing.assert_close(
        matrices.absolute_activation_error,
        (matrices.reference_activation -
         matrices.quantized_activation).abs())
    torch.testing.assert_close(
        matrices.absolute_weight_error,
        (matrices.reference_weight - matrices.quantized_weight).abs())


def test_capture_round_trips_native_fp32_tensors(tmp_path: Path):
    capture, _, _, _, _ = _capture()
    path = tmp_path / "capture.npz"

    capture.save(path)
    loaded = ConvMatrixCapture.load(path)

    assert loaded.module == capture.module
    assert loaded.sample_index == capture.sample_index
    assert loaded.geometry == capture.geometry
    assert loaded.activation_bits == 8
    assert loaded.activation_unsigned is False
    torch.testing.assert_close(loaded.reference_input, capture.reference_input)
    torch.testing.assert_close(loaded.quantized_input, capture.quantized_input)
    torch.testing.assert_close(loaded.original_weight, capture.original_weight)
    torch.testing.assert_close(
        loaded.quantized_weight, capture.quantized_weight)
    torch.testing.assert_close(
        loaded.activation_scale, capture.activation_scale)


def test_capture_rejects_nonfinite_or_non_fp32_tensors():
    capture, layer, inputs, quantized_inputs, quantized_weight = _capture()
    del capture
    bad = inputs.clone()
    bad[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        ConvMatrixCapture.from_tensors(
            "conv", 7, layer, bad, quantized_inputs,
            layer.weight.detach(), quantized_weight,
            8, False, torch.tensor(0.125))
    with pytest.raises(TypeError, match="float32"):
        ConvMatrixCapture.from_tensors(
            "conv", 7, layer, inputs.double(), quantized_inputs.double(),
            layer.weight.detach().double(), quantized_weight.double(),
            8, False, torch.tensor(0.125, dtype=torch.float64))


@pytest.mark.parametrize("shape", ((3, 5), (5, 3)))
def test_full_line_curtain_preserves_every_matrix_element(shape):
    matrix = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    curtain = full_line_curtain(matrix)

    assert curtain.rendered_elements == matrix.size
    assert len(curtain.lines) == min(shape)
    assert curtain.orientation == (
        "rows_along_x" if shape[0] <= shape[1] else "columns_along_y")
    np.testing.assert_array_equal(curtain.reconstruct(), matrix)


def test_full_line_curtain_rejects_nonfinite_or_non_fp32_matrix():
    with pytest.raises(TypeError, match="float32"):
        full_line_curtain(np.ones((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        full_line_curtain(np.asarray([[1.0, np.inf]], dtype=np.float32))


def test_recorder_captures_only_declared_module_sample_pairs():
    first = nn.Conv2d(1, 2, kernel_size=1, bias=False)
    second = nn.Conv2d(2, 1, kernel_size=1, bias=False)
    modules = {"first": first, "second": second}
    originals = {
        "first": first.weight.detach().cpu().clone(),
        "second": second.weight.detach().cpu().clone(),
    }
    recorder = FullMatrixCaptureRecorder(
        modules, originals, {7: {"first"}, 9: {"second"}})
    quantizer = UniformQuantizer(0.125)
    reference = torch.ones(1, 1, 2, 2)
    quantized = torch.full_like(reference, 0.875)
    codes = torch.full_like(reference, 7, dtype=torch.int32)

    recorder.begin_sample(7)
    recorder.record(
        "first", "input", 0, "encoder", reference, quantized,
        codes, quantizer, 1)
    captures = recorder.end_sample()

    assert set(captures) == {"first"}
    assert captures["first"].sample_index == 7
    recorder.clear_sample()


def test_recorder_rejects_missing_selected_input():
    layer = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    recorder = FullMatrixCaptureRecorder(
        {"conv": layer}, {"conv": layer.weight.detach().cpu().clone()},
        {7: {"conv"}})
    recorder.begin_sample(7)

    with pytest.raises(RuntimeError, match="missing"):
        recorder.end_sample()
