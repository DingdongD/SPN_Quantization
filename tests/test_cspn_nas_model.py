from pathlib import Path
import sys

import torch

from spn_quant.nas.spec import EncoderSpec


MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from cspn_encoder_nas import build_cspn_nas  # noqa: E402


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def test_reduced_candidate_preserves_encoder_decoder_contract():
    spec = EncoderSpec(
        stem_width=32,
        widths=(32, 64, 128, 256),
        depths=(1, 1, 1, 0),
    )
    model = build_cspn_nas(spec, cspn_step=1)
    model.eval()
    inputs = torch.rand(1, 4, 228, 304)

    with torch.inference_mode():
        features = model.forward_encoder(inputs)
        output = model(inputs)

    assert [tuple(value.shape) for value in features] == [
        (1, 64, 114, 152),
        (1, 64, 57, 76),
        (1, 128, 29, 38),
        (1, 512, 8, 10),
    ]
    assert model.decoder_channels == (64, 64, 128, 512)
    assert output.shape == (1, 1, 228, 304)
    assert torch.isfinite(output).all()


def test_reduced_candidate_has_fewer_parameters_than_r18_control():
    reduced = build_cspn_nas(
        EncoderSpec(32, (32, 64, 128, 256), (1, 1, 1, 0)),
        cspn_step=1,
    )
    control = build_cspn_nas(EncoderSpec.r18(), cspn_step=1)

    assert _parameter_count(reduced) < _parameter_count(control)
