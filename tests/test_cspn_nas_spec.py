import pytest

from spn_quant.nas.spec import DecoderSpec, EncoderSpec, enumerate_encoder_specs


def test_r18_spec_has_stable_identity():
    spec = EncoderSpec.r18()

    assert spec.depths == (2, 2, 2, 2)
    assert spec.widths == (64, 128, 256, 512)
    assert spec.slug == "s64-w64-128-256-512-d2-2-2-2"
    assert EncoderSpec.from_dict(spec.to_dict()) == spec


def test_projection_mode_is_canonical_depth_zero():
    spec = EncoderSpec(
        stem_width=32,
        widths=(32, 64, 128, 256),
        depths=(1, 1, 1, 0),
    )

    assert spec.depths[-1] == 0


def test_invalid_widths_are_rejected():
    with pytest.raises(ValueError, match="non-decreasing"):
        EncoderSpec(
            stem_width=32,
            widths=(32, 64, 48, 128),
            depths=(1, 1, 1, 1),
        )


def test_enumeration_is_deterministic_unique_and_contains_control():
    first = list(enumerate_encoder_specs())
    second = list(enumerate_encoder_specs())

    assert first == second
    assert len(first) == len(set(first))
    assert EncoderSpec.r18() in first
    assert any(spec.depths[3] == 0 for spec in first)
    assert all(width % 16 == 0 for spec in first for width in spec.widths)


def test_decoder_spec_has_stable_default_and_scaled_identity():
    control = DecoderSpec.default()
    half = DecoderSpec.scaled(0.5)

    assert control.widths == (512, 256, 128, 64, 64)
    assert half.widths == (256, 128, 64, 32, 32)
    assert half.slug == "dw256-128-64-32-32"
    assert DecoderSpec.from_dict(half.to_dict()) == half


def test_decoder_widths_must_be_nonincreasing_hardware_multiples():
    with pytest.raises(ValueError, match="non-increasing"):
        DecoderSpec((256, 128, 64, 32, 48))

    with pytest.raises(ValueError, match="multiples of 8"):
        DecoderSpec((256, 128, 64, 32, 28))
