from pathlib import Path

import pytest

from scripts import build_rk3588_precision_variants as builder


def test_precision_plan_distinguishes_fp16_and_native_w8a8():
    plan = builder.precision_plan()

    assert tuple(item.name for item in plan) == ("fp16", "w8a8")
    assert plan[0].config == {"float_dtype": "float16"}
    assert plan[0].quantize is False
    assert plan[1].config == {
        "quantized_dtype": "w8a8",
        "quantized_algorithm": "normal",
        "quantized_method": "channel",
    }
    assert plan[1].quantize is True


def test_w8a8_requires_a_real_calibration_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        builder.validate_calibration(tmp_path / "missing.txt", 128)

    dataset = tmp_path / "dataset.txt"
    dataset.write_text("a.npy b.npy\n" * 127, encoding="utf-8")
    with pytest.raises(ValueError):
        builder.validate_calibration(dataset, 128)

    dataset.write_text("a.npy b.npy\n" * 128, encoding="utf-8")
    assert builder.validate_calibration(dataset, 128) == dataset.resolve()
