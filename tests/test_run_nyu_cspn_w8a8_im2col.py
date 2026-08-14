import json
from pathlib import Path

import pytest

from scripts import run_nyu_cspn_w8a8_im2col as runner


class TestCSPNW8A8Im2ColRunner:
    @staticmethod
    def _metadata(root: Path):
        checkpoint = root / "best.pt"
        checkpoint.write_bytes(b"checkpoint")
        metadata = {
            "model": "cspn",
            "checkpoint": str(checkpoint),
            "seed": 71,
            "calibration_indices": list(range(128)),
            "evaluation_indices": list(range(200, 264)),
        }
        path = root / "metadata.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return checkpoint, path

    def test_protocol_requires_declared_checkpoint_and_fixed_sample_counts(
            self, tmp_path):
        checkpoint, metadata = self._metadata(tmp_path)

        protocol = runner.load_protocol(checkpoint, metadata)

        assert protocol["seed"] == 71
        assert len(protocol["calibration_indices"]) == 128
        assert len(protocol["evaluation_indices"]) == 64

    def test_protocol_rejects_duplicate_indices(self, tmp_path):
        checkpoint, metadata = self._metadata(tmp_path)
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["calibration_indices"][-1] = 0
        metadata.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError):
            runner.load_protocol(checkpoint, metadata)

    def test_selects_exact_propagation_aware_w8a8_contract(self):
        config = runner.select_w8a8_configuration(
            ("encoder", "decoder", "depth_head"))

        assert config["name"] == "PA_W8A8"
        assert (config["w_bits"], config["a_bits"]) == (8, 8)
        assert config["propagation"] == {
            "affinity_bits": 8,
            "confidence_bits": 8,
            "offset_bits": 8,
            "state_bits": 8,
            "coefficient_fraction_bits": 13,
        }

    def test_rejects_modified_w8a8_contract(self):
        config = runner.select_w8a8_configuration(
            ("encoder", "decoder", "depth_head"))
        config["a_bits"] = 4

        with pytest.raises(ValueError):
            runner.validate_w8a8_configuration(config)

    def test_cli_requires_all_runtime_and_memory_limits(self):
        args = runner.parse_args([
            "--device", "cuda:0",
            "--checkpoint", "/repo/best.pt",
            "--data-root", "/dataset",
            "--stratified-metadata", "/metrics/metadata.json",
            "--output-dir", "/metrics/im2col",
            "--percentile-capacity", "128",
            "--token-topk", "16",
            "--token-chunk", "256",
            "--plot-layer-count", "6",
            "--plot-sample-count", "3",
            "--fold-max-error", "0.00001",
        ])

        assert args.device == "cuda:0"
        assert args.token_chunk == 256
        assert args.fold_max_error == 0.00001
