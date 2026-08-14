import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
import numpy as np

from scripts import plot_nyu_cspn_w8a8_im2col as plotter


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
        "selected_plot_modules": ["conv"],
        "selected_plot_samples": [7],
    }), encoding="utf-8")
    rows = []
    for channel in range(3):
        for offset in range(2):
            rows.append({
                "module": "conv",
                "channel": channel,
                "kernel_row": 0,
                "kernel_col": offset,
                "kernel_offset": offset,
                "activation_rms": 0.1 + channel + offset,
                "activation_p99": 0.2 + channel + offset,
                "activation_sqnr_db": 30.0 - channel - offset,
                "activation_new_zero_rate": 0.01 * (channel + offset),
                "weight_rms": 0.3 + channel + offset,
                "weight_sqnr_db": 40.0 - channel - offset,
            })
    _write_csv(root / "channel_offset_metrics.csv", rows)
    spatial = root / "spatial_tokens" / "conv"
    spatial.mkdir(parents=True)
    grid = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.savez_compressed(
        spatial / "sample_00007.npz",
        module=np.array("conv"), sample_index=np.array(7),
        patch_rms=grid + 1.0,
        patch_p99=grid + 2.0,
        patch_maximum_abs=grid + 3.0,
        activation_error=(grid + 1.0) ** 2,
        activation_new_zero_count=grid.astype(np.int64),
        activation_saturation_count=(grid / 2).astype(np.int64),
        local_output_error=(grid + 1.0) ** 3,
    )
    _write_csv(root / "spatial_manifest.csv", [{
        "module": "conv",
        "sample_index": 7,
        "path": "spatial_tokens/conv/sample_00007.npz",
    }])


def test_style_uses_arial_first():
    plotter.set_style(12)

    assert plotter.plt.rcParams["font.sans-serif"][0] == "Arial"
    assert plotter.plt.rcParams["font.size"] == 12


def test_plotter_generates_nonblank_k_axis_and_spatial_figures(tmp_path):
    experiment = tmp_path / "experiment"
    output = tmp_path / "figures"
    _experiment(experiment)

    completed = subprocess.run([
        sys.executable, "scripts/plot_nyu_cspn_w8a8_im2col.py",
        "--experiment-dir", str(experiment),
        "--output-dir", str(output),
        "--font-size", "11",
        "--dpi", "80",
        "--row-stride", "1",
    ], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    expected = (
        output / "k_axis_3d" / "conv" / "distribution.png",
        output / "k_axis_3d" / "conv" / "distribution.pdf",
        output / "spatial_3d" / "conv" / "sample_00007.png",
        output / "spatial_3d" / "conv" / "sample_00007.pdf",
    )
    assert all(path.is_file() for path in expected)
    for path in (expected[0], expected[2]):
        image = mpimg.imread(path)
        assert float(image.std()) > 0.01
    with (output / "figure_manifest.csv").open(
            newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["kind"], row["module"]) for row in rows} == {
        ("k_axis_3d", "conv"), ("spatial_3d", "conv")}


def test_plot_script_help_runs_from_repository_root():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/plot_nyu_cspn_w8a8_im2col.py", "--help"],
        cwd=repository, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
