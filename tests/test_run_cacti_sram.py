import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_cacti_sram import cacti_cli_arguments, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload():
    return {
        "format_version": 1,
        "capacity_bytes": 4 * 1024 * 1024,
        "block_bytes": 64,
        "bank_count": 16,
        "technology_nm": 32,
        "output_width_bits": 512,
        "temperature_k": 300,
    }


def test_cacti_cli_uses_fully_initialized_63_argument_interface(tmp_path):
    path = tmp_path / "sram.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    config = load_config(path)
    arguments = cacti_cli_arguments(config)

    assert len(arguments) == 63
    assert arguments[0] == str(4 * 1024 * 1024)
    assert arguments[8] == "16"
    assert arguments[9] == "32"
    assert arguments[10] == "512"
    assert arguments[51] == "0"  # no ECC storage overhead
    assert arguments[52] == "0"  # model a 2D SRAM, not 3D DRAM


def test_cacti_config_rejects_non_power_of_two_capacity(tmp_path):
    payload = _payload()
    payload["capacity_bytes"] -= 1
    path = tmp_path / "sram.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="power of two"):
        load_config(path)


def test_cacti_config_rejects_non_byte_aligned_port(tmp_path):
    payload = _payload()
    payload["output_width_bits"] = 510
    path = tmp_path / "sram.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="width"):
        load_config(path)


def test_compatibility_patch_applies_to_pinned_cacti_source():
    completed = subprocess.run(
        ("patch", "--dry-run", "-p1", "-i", str(
            REPO_ROOT / "patches" / "cacti" /
            "0001-initialize-input-parameters.patch")),
        cwd=str(REPO_ROOT / "external" / "cacti"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert completed.returncode == 0, completed.stdout
