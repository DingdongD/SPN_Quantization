from pathlib import Path

import pytest

from spn_quant.qat.method_config import load_method_config


CONFIG = Path(__file__).resolve().parents[1] / \
    "configs/cspn_lsqplus_hawq.json"


def test_formal_config_declares_exact_methods_and_gpu_owners():
    config = load_method_config(CONFIG)

    assert config.model == "cspn"
    assert config.lsqplus.bits == (4, 6)
    assert config.hawq.bits == (4, 6, 8)
    assert config.hawq.maximum_average_weight_bits == 6.0
    assert config.hawq.maximum_average_activation_bits == 6.0
    assert config.gpus == (
        ("baselines", 0),
        ("lsqplus_w4a4", 1),
        ("lsqplus_w6a6", 2),
        ("hawq_mixed_le6", 3),
    )


def test_missing_required_field_is_not_defaulted(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"model": "cspn"}', encoding="utf-8")

    with pytest.raises(KeyError):
        load_method_config(path)
