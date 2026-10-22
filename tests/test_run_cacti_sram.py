import json
from pathlib import Path
import subprocess

import pytest

from scripts.run_cacti_sram import cacti_cli_arguments, load_config, run_many


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


def test_cacti_sweep_builds_once_for_multiple_macros(tmp_path, monkeypatch):
    first_payload = _payload()
    second_payload = _payload()
    second_payload["capacity_bytes"] = 8 * 1024 * 1024
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(first_payload), encoding="utf-8")
    second.write_text(json.dumps(second_payload), encoding="utf-8")
    calls = {"build": 0, "execute": 0}

    def fake_build(build_root):
        calls["build"] += 1
        return build_root / "cacti"

    def fake_execute(executable, config):
        calls["execute"] += 1
        return """
Cache size                    : %d
Cache banks (UCA)             : %d
Access time (ns): 1.25
Cycle time (ns): 0.80
Total dynamic read energy per access (nJ): 2.50
Total dynamic write energy per access (nJ): 2.75
Cache height x width (mm): 4.0 x 5.0
""" % (config.capacity_bytes, config.bank_count)

    monkeypatch.setattr("scripts.run_cacti_sram._build_executable", fake_build)
    monkeypatch.setattr("scripts.run_cacti_sram._execute", fake_execute)
    monkeypatch.setattr(
        "scripts.run_cacti_sram._git_revision", lambda root:
        "1ffd8dfb10303d306ecd8d215320aea07651e878")
    jobs = tuple(
        (config, tmp_path / (name + ".txt"), tmp_path / (name + ".result.json"))
        for config, name in ((first, "first"), (second, "second")))

    outputs = run_many(jobs)

    assert calls == {"build": 1, "execute": 2}
    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    assert json.loads(outputs[1].read_text(encoding="utf-8"))["result"][
        "capacity_bytes"] == 8 * 1024 * 1024
