import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is unavailable")
def test_bilinear_grid_sample_matches_align_corners_false_zero_padding(tmp_path):
    source = tmp_path / "grid_sample_test.cpp"
    binary = tmp_path / "grid_sample_test"
    source.write_text(
        r'''#include <cmath>
#include <iostream>
#include "tools/rknn_grid_sample.h"

int main() {
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float grid[] = {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
  float output[3] = {};
  spn_rknn::grid_sample_bilinear_zeros_align_corners_false_nchw(
      input, grid, output, 1, 1, 2, 2, 1, 3, false);
  const float expected[] = {0.25f, 2.5f, 1.0f};
  for (int i = 0; i < 3; ++i) {
    if (std::abs(output[i] - expected[i]) > 1.0e-6f) return i + 1;
  }
  std::cout << output[0] << "," << output[1] << "," << output[2] << "\n";
  return 0;
}
''',
        encoding="utf-8")

    compile_result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I", str(REPO_ROOT),
         str(source), "-o", str(binary)],
        check=False, capture_output=True, text=True)
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr
    assert run_result.stdout.strip() == "0.25,2.5,1"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is unavailable")
def test_grid_layout_can_be_nchw(tmp_path):
    source = tmp_path / "grid_layout_test.cpp"
    binary = tmp_path / "grid_layout_test"
    source.write_text(
        r'''#include <cmath>
#include "tools/rknn_grid_sample.h"

int main() {
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float grid_nchw[] = {-1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f};
  float output[3] = {};
  spn_rknn::grid_sample_bilinear_zeros_align_corners_false_nchw(
      input, grid_nchw, output, 1, 1, 2, 2, 1, 3, true);
  const float expected[] = {0.25f, 2.5f, 1.0f};
  for (int i = 0; i < 3; ++i) {
    if (std::abs(output[i] - expected[i]) > 1.0e-6f) return i + 1;
  }
  return 0;
}
''',
        encoding="utf-8")
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I", str(REPO_ROOT),
         str(source), "-o", str(binary)],
        check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(binary)], check=False).returncode == 0


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is unavailable")
def test_nhwc_feature_and_output_layout(tmp_path):
    source = tmp_path / "nhwc_test.cpp"
    binary = tmp_path / "nhwc_test"
    source.write_text(
        r'''#include <cmath>
#include "tools/rknn_grid_sample.h"

int main() {
  const float input[] = {1, 10, 2, 20, 3, 30, 4, 40};
  const float grid[] = {0, 0};
  float output[2] = {};
  spn_rknn::grid_sample_bilinear_zeros_align_corners_false(
      input, grid, output, 1, 2, 2, 2, 1, 1,
      true, false, true);
  return (std::abs(output[0] - 2.5f) < 1e-6f &&
          std::abs(output[1] - 25.0f) < 1e-6f) ? 0 : 1;
}
''',
        encoding="utf-8")
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I", str(REPO_ROOT),
         str(source), "-o", str(binary)],
        check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert subprocess.run([str(binary)], check=False).returncode == 0
