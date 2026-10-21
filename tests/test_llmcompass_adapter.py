from pathlib import Path

from spn_quant.hardware.llmcompass_adapter import ArrayShape, simulate_gemm
from spn_quant.hardware.workload import GemmWorkload


def test_llmcompass_submodule_and_dependency_are_pinned():
    root = Path(__file__).resolve().parents[1]
    modules = (root / ".gitmodules").read_text(encoding="utf-8")
    requirements = (root / "requirements-llmcompass.txt").read_text(
        encoding="utf-8")
    assert 'path = external/LLMCompass' in modules
    assert 'branch = ISCA_AE' in modules
    assert requirements.strip() == "scalesim==2.0.2"


def test_llmcompass_cycle_result_has_bounded_effective_utilization():
    result = simulate_gemm(
        GemmWorkload("x", "conv2d", 1024, 64, 288, 1),
        ArrayShape(16, 16), mac_per_cycle=1, dataflow="os")
    assert result.backend == "llmcompass"
    assert result.simulation_source == "analytical"
    assert result.cycles > 0
    assert 0 < result.effective_utilization <= 1
    assert result.macs == 1024 * 64 * 288


def test_narrow_output_exposes_wide_array_underutilization():
    work = GemmWorkload("head", "conv2d", 4096, 1, 64, 1)
    narrow = simulate_gemm(work, ArrayShape(16, 8), 1, "os")
    wide = simulate_gemm(work, ArrayShape(8, 16), 1, "os")
    assert narrow.effective_utilization > wide.effective_utilization


def test_repetitions_scale_cycles_and_macs_without_changing_utilization():
    single = simulate_gemm(
        GemmWorkload("group", "conv2d", 1024, 16, 72, 1),
        ArrayShape(16, 16), 1, "os")
    repeated = simulate_gemm(
        GemmWorkload("group", "conv2d", 1024, 16, 72, 4),
        ArrayShape(16, 16), 1, "os")
    assert repeated.cycles == 4 * single.cycles
    assert repeated.macs == 4 * single.macs
    assert repeated.effective_utilization == single.effective_utilization
