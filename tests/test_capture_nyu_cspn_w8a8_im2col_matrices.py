import csv
import json
from pathlib import Path

import pytest

from scripts import capture_nyu_cspn_w8a8_im2col_matrices as capture


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _experiment(root: Path):
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({
        "model": "cspn",
        "configuration": "PA_W8A8",
        "selected_plot_modules": ["conv", "decoder.conv"],
    }), encoding="utf-8")
    _write_csv(root / "top_spatial_tokens.csv", [
        {"module": "conv", "sample_index": 9,
         "local_output_error": 2.0},
        {"module": "conv", "sample_index": 7,
         "local_output_error": 2.0},
        {"module": "decoder.conv", "sample_index": 11,
         "local_output_error": 3.0},
        {"module": "other", "sample_index": 5,
         "local_output_error": 10.0},
    ])


def test_select_worst_samples_uses_ranked_modules_and_stable_ties(tmp_path):
    experiment = tmp_path / "experiment"
    _experiment(experiment)

    selected = capture.select_worst_samples(experiment)

    assert selected == {"conv": 7, "decoder.conv": 11}


def test_select_worst_samples_rejects_missing_ranked_module(tmp_path):
    experiment = tmp_path / "experiment"
    _experiment(experiment)
    rows = [{
        "module": "conv", "sample_index": 7,
        "local_output_error": 2.0,
    }]
    _write_csv(experiment / "top_spatial_tokens.csv", rows)

    with pytest.raises(ValueError, match="coverage"):
        capture.select_worst_samples(experiment)


def test_parse_args_requires_experiment_device_and_fold_threshold():
    with pytest.raises(SystemExit):
        capture.parse_args([])
    args = capture.parse_args([
        "--experiment-dir", "/experiment",
        "--device", "cuda:0",
        "--fold-max-error", "0.05",
    ])
    assert args.experiment_dir == "/experiment"
    assert args.device == "cuda:0"
    assert args.fold_max_error == 0.05
