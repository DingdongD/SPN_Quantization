from pathlib import Path

import numpy as np
import pytest

from scripts import plot_nyu_cspn_group_a4_qat as plotter


def _payload(path: Path, config: str, nonfinite: bool = False):
    pred = np.ones((3, 4), dtype=np.float32)
    if nonfinite:
        pred[0, 0] = np.nan
    np.savez_compressed(
        path,
        gt=np.ones((3, 4), dtype=np.float32),
        fp32=np.ones((3, 4), dtype=np.float32),
        pred=pred,
        abs_err=np.zeros((3, 4), dtype=np.float32),
        valid_gt=np.ones((3, 4), dtype=np.bool_),
        sparse=np.zeros((3, 4), dtype=np.float32),
        rgb=np.ones((3, 4, 3), dtype=np.float32),
        sample_index=np.array(7),
        model=np.array("cspn"),
        config=np.array(config),
    )


def test_plot_loader_rejects_nonfinite_payload(tmp_path):
    path = tmp_path / "sample.npz"
    _payload(path, "QAT_STATIC_G8_W4A4", nonfinite=True)

    with pytest.raises(ValueError, match="non-finite"):
        plotter.load_payload(path, "QAT_STATIC_G8_W4A4")


def test_plot_loader_requires_matching_configuration(tmp_path):
    path = tmp_path / "sample.npz"
    _payload(path, "QAT_STATIC_G8_W4A4")

    with pytest.raises(ValueError, match="configuration"):
        plotter.load_payload(path, "QAT_DYNAMIC_G8_W4A4")


def test_style_uses_arial_first():
    plotter.set_style(13)

    assert plotter.plt.rcParams["font.sans-serif"][0] == "Arial"
    assert plotter.plt.rcParams["font.size"] == 13
