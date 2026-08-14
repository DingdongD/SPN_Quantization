import csv
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
import torch
import torch.nn as nn

from scripts import plot_nyu_cspn_w8a8_im2col_matrices as plotter
from spn_quant.im2col_matrix_visualization import ConvMatrixCapture


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _capture_root(root: Path):
    root.mkdir()
    directory = root / "captures" / "conv"
    directory.mkdir(parents=True)
    layer = nn.Conv2d(2, 3, kernel_size=(2, 2), padding=1, bias=False)
    reference = torch.arange(
        1 * 2 * 4 * 5, dtype=torch.float32).reshape(1, 2, 4, 5) / 10.0
    quantized = torch.round(reference / 0.25) * 0.25
    quantized_weight = torch.round(layer.weight.detach() / 0.02) * 0.02
    capture = ConvMatrixCapture.from_tensors(
        "conv", 7, layer, reference, quantized,
        layer.weight.detach(), quantized_weight,
        8, False, torch.tensor(0.25, dtype=torch.float32))
    path = directory / "sample_00007.npz"
    capture.save(path)
    matrices = capture.matrices()
    _write_csv(root / "capture_manifest.csv", [{
        "module": "conv",
        "sample_index": 7,
        "path": str(path.relative_to(root)),
        "input_shape": "1x2x4x5",
        "weight_shape": "3x2x2x2",
        "k_size": matrices.reference_weight.shape[1],
        "token_count": matrices.reference_activation.shape[0],
        "activation_bits": 8,
        "activation_unsigned": 0,
        "activation_scale_count": 1,
    }])
    return matrices


def test_plotter_writes_full_unsampled_weight_and_activation_triptychs(
        tmp_path):
    capture_root = tmp_path / "capture"
    output = tmp_path / "figures"
    matrices = _capture_root(capture_root)

    completed = subprocess.run([
        sys.executable,
        "scripts/plot_nyu_cspn_w8a8_im2col_matrices.py",
        "--capture-dir", str(capture_root),
        "--output-dir", str(output),
        "--font-size", "10",
        "--dpi", "70",
        "--line-width", "0.5",
        "--elevation", "25",
        "--azimuth", "-58",
    ], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    expected = (
        output / "conv" / "sample_00007_weights.png",
        output / "conv" / "sample_00007_weights.pdf",
        output / "conv" / "sample_00007_activations.png",
        output / "conv" / "sample_00007_activations.pdf",
    )
    assert all(path.is_file() for path in expected)
    for path in (expected[0], expected[2]):
        image = mpimg.imread(path)
        assert float(image.std()) > 0.01
    with (output / "figure_manifest.csv").open(
            newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["kind"] for row in rows} == {"weight", "activation"}
    for row in rows:
        assert row["sampling"] == "none"
        assert int(row["rendered_elements"]) == int(row["matrix_elements"])
        assert float(row["shared_reference_qdq_maximum"]) > 0.0
        assert float(row["error_maximum"]) >= 0.0
    weight = next(row for row in rows if row["kind"] == "weight")
    activation = next(row for row in rows if row["kind"] == "activation")
    assert int(weight["matrix_rows"]) == \
        matrices.reference_weight.shape[0]
    assert int(weight["matrix_columns"]) == \
        matrices.reference_weight.shape[1]
    assert int(activation["matrix_rows"]) == \
        matrices.reference_activation.shape[0]
    assert int(activation["matrix_columns"]) == \
        matrices.reference_activation.shape[1]


def test_plot_script_help_runs_from_repository_root():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run([
        sys.executable,
        "scripts/plot_nyu_cspn_w8a8_im2col_matrices.py", "--help",
    ], cwd=repository, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_zero_error_panel_is_explicitly_labelled():
    assert plotter._panel_title("Absolute A8 error", 0.0) == \
        "Absolute A8 error (all zero)"
    assert plotter._panel_title("Absolute A8 error", 0.25) == \
        "Absolute A8 error"
