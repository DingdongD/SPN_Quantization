# Four-SPN Cross-Platform Energy Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure auditable per-forward energy for the four vanilla SPN models on the existing RTX 3080 Ti, Jetson AGX Xavier, and RK3588 deployments, and add the results beside the latency data in the existing experiment report.

**Architecture:** Reuse the exact fixed input, checkpoint, precision, complete-forward boundary, clocks, affinity, and model implementations from the latency experiment. Measure long continuous inference windows after warmup, sample hardware power counters concurrently, integrate timestamped power with the trapezoidal rule, and report both gross energy and idle-subtracted incremental energy per completed forward. Reject platforms without a real current/power sensor rather than substituting TDP, DVFS voltage, or a software estimate.

**Tech Stack:** Python 3.7+, PyTorch/CUDA, NVML, jetson-stats/jtop with Xavier INA3221 telemetry, RKNN C++ runtime, pytest, JSON/Markdown evidence.

---

### Task 1: Energy integration and statistics

**Files:**
- Create: `spn_quant/energy.py`
- Create: `tests/test_energy.py`

- [ ] **Step 1: Write failing integration tests**

```python
def test_integrate_power_clips_and_interpolates_window_edges():
    samples = [(0.0, 10.0), (1.0, 20.0), (2.0, 10.0)]
    assert integrate_power_joules(samples, 0.5, 1.5) == pytest.approx(17.5)

def test_energy_per_forward_reports_gross_and_idle_subtracted_values():
    result = energy_per_forward(
        active_energy_j=200.0, active_seconds=10.0,
        idle_mean_power_w=5.0, completed_forwards=100)
    assert result["gross_j_per_forward"] == pytest.approx(2.0)
    assert result["incremental_j_per_forward"] == pytest.approx(1.5)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_energy.py`

Expected: collection fails because `spn_quant.energy` does not exist.

- [ ] **Step 3: Implement clipped trapezoidal integration and validation**

Implement `integrate_power_joules`, `summarize_power_window`, and `energy_per_forward`. Reject non-monotonic timestamps, missing edge coverage, non-positive durations/counts, non-finite readings, and negative incremental energy.

- [ ] **Step 4: Run focused and baseline tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_energy.py tests/test_benchmark_nyu_vanilla_gpu_worker.py tests/test_summarize_spn_cross_platform.py`

Expected: all tests pass.

### Task 2: Hardware power samplers and complete-forward energy worker

**Files:**
- Modify: `spn_quant/energy.py`
- Modify: `scripts/benchmark_nyu_vanilla_gpu_worker.py`
- Modify: `tests/test_energy.py`
- Modify: `tests/test_benchmark_nyu_vanilla_gpu_worker.py`

- [ ] **Step 1: Write failing sampler and phase-contract tests**

```python
def test_power_sample_schema_preserves_source_and_rails():
    sample = normalize_power_sample(1_000_000_000, {"board": 12_000})
    assert sample == {"monotonic_ns": 1_000_000_000,
                      "power_w": {"board": 12.0}}

def test_energy_contract_requires_synchronized_full_forwards():
    assert validate_energy_settings({
        "warmup_iterations": 30, "idle_seconds": 15,
        "active_seconds": 60, "sample_interval_ms": 50,
        "synchronize_each_forward": True}) is None
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_energy.py tests/test_benchmark_nyu_vanilla_gpu_worker.py`

Expected: tests fail because sampler normalization and the energy phase do not exist.

- [ ] **Step 3: Implement NVML and jtop sampler adapters**

The NVML adapter reads GPU board power through `nvmlDeviceGetPowerUsage`. The jtop adapter records `Power TOT`, `Power GPU`, `Power SOC`, `Power CPU`, `Power CV`, `Power VDDRQ`, and `Power SYS5V` from the root-owned INA3221 service without changing sensor configuration.

- [ ] **Step 4: Implement the controlled energy phase**

Add optional CLI arguments:

```text
--energy-output PATH
--power-source nvml|jtop
--energy-idle-seconds 15
--energy-active-seconds 60
--power-sample-interval-ms 50
```

After model validation and warmup, synchronize CUDA, collect an idle window with the loaded model and resident inputs, then execute synchronized complete forwards until the active duration is met. Record raw timestamped samples, exact phase boundaries, completed-forward count, wall duration, CUDA-event latency statistics, gross energy, incremental energy, clocks, temperature, and sensor scope.

- [ ] **Step 5: Run focused and baseline tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_energy.py tests/test_benchmark_nyu_vanilla_gpu_worker.py tests/test_run_nyu_vanilla_gpu_latency.py`

Expected: all tests pass.

### Task 3: RTX 3080 Ti controlled measurements

**Files:**
- Create: `reports/generated/spn_cross_platform_20261022/autodl_nvidia/energy/controlled_repeats/*.json`
- Create: `reports/generated/spn_cross_platform_20261022/autodl_nvidia/energy/inventory.json`

- [ ] **Step 1: Deploy the tested worker without replacing model artifacts**

Copy only the new worker and `spn_quant/energy.py` into the existing `/root/autodl-tmp/spn_cross_platform_20261022` deployment and verify their SHA-256 hashes.

- [ ] **Step 2: Verify an idle GPU and run one smoke measurement**

Require GPU utilization at or below 5% before loading the model. Run CSPN with 30 warmups, a 5-second idle window, and a 10-second active window; reject missing samples or non-finite output.

- [ ] **Step 3: Run five independent measurements per model**

For CSPN, DySPN, NLSPN, and CompletionFormer, run five fresh processes with 30 warmups, 15 seconds idle, 60 seconds active, FP32, TF32 disabled, fixed sample 386, and the existing GPU-resident complete-forward boundary.

- [ ] **Step 4: Retrieve raw evidence and validate it locally**

Require at least 100 samples in each idle and active window, positive completed-forward counts, identical checkpoint/input hashes, and no overlapping foreign GPU process.

### Task 4: Jetson AGX Xavier controlled measurements

**Files:**
- Create: `reports/generated/spn_cross_platform_20261022/jetson_xavier/energy/controlled_repeats/*.json`
- Create: `reports/generated/spn_cross_platform_20261022/jetson_xavier/energy/inventory.json`

- [ ] **Step 1: Verify INA3221 access through jtop**

Require `jtop.service` active and verify non-zero `TOT`, `GPU`, and `SOC` readings. Record rail labels and the fact that total-module and individual rails have different measurement boundaries.

- [ ] **Step 2: Deploy and smoke-test the energy worker**

Use the existing Xavier source tree, FP32 checkpoints, sample 386, CUDA environment, CPU7 affinity, MAXN mode, fixed 1377 MHz GPU clock, and jtop power source.

- [ ] **Step 3: Run five independent measurements per model**

Use the same 30-warmup, 15-second idle, 60-second active, 50-ms requested sampling protocol as RTX. Preserve all raw rail samples.

- [ ] **Step 4: Retrieve and validate raw evidence**

Reject runs with changed clocks, missing INA rails, non-finite output, fewer than 100 samples per phase, or background GPU activity outside the benchmark.

### Task 5: RK3588 measurement capability gate

**Files:**
- Create: `reports/generated/spn_cross_platform_20261022/rk3588/energy/capability_audit.json`
- Create when a meter exists: `reports/generated/spn_cross_platform_20261022/rk3588/energy/controlled_repeats/*.json`

- [ ] **Step 1: Record the on-board sensor audit**

Record all hwmon, IIO, power-supply, RKNPU debug, and device-tree interfaces. Explicitly classify temperature, utilization, frequency, and voltage-only interfaces as insufficient for measured energy because current is unavailable.

- [ ] **Step 2: Admit only a physical input-power/current sensor**

Accept an external USB-C/5-V shunt meter only if it exports timestamped voltage/current or power samples at a documented interval. Do not derive energy from NPU TOPS, TDP, operating voltage, or a fitted software model.

- [ ] **Step 3: If admitted, run five independent measurements per model**

Reuse the exact RKNN FP16 model/input hashes, AUTO core mask, CPU6-7 affinity, OMP=2, 1-GHz NPU and 2.4-GHz DMC settings. Use loaded-runtime idle and 60-second synchronous full-forward active windows, including CPU custom operators and NPU/CPU transfers.

### Task 6: Aggregate statistics and update the experiment record

**Files:**
- Create: `scripts/summarize_spn_cross_platform_energy.py`
- Create: `tests/test_summarize_spn_cross_platform_energy.py`
- Modify: `reports/generated/spn_cross_platform_20261022/EXPERIMENT.md`
- Modify: `reports/generated/spn_cross_platform_20261022/manifest.json`

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_summary_uses_independent_run_means_for_ci():
    summary = summarize_runs(FIVE_VALID_RUNS)
    assert summary["process_count"] == 5
    assert summary["ci95_high_j"] > summary["ci95_low_j"]

def test_summary_rejects_mixed_sensor_boundaries():
    with pytest.raises(ValueError, match="sensor scope"):
        summarize_runs(MIXED_BOARD_AND_RAIL_RUNS)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_summarize_spn_cross_platform_energy.py`

Expected: collection fails because the summarizer does not exist.

- [ ] **Step 3: Implement aggregation and evidence hashing**

Compute mean, median, range, sample CV, and two-sided Student-t 95% confidence intervals over five independent process-level energy values. Preserve gross and incremental energy separately and never combine unlike sensor boundaries in one comparison column.

- [ ] **Step 4: Update the Markdown report**

Add protocol, sensor boundaries, power and energy tables, uncertainty, raw-evidence links, and a prominent unavailable result for any platform that fails the physical-sensor gate. Do not present unavailable RK3588 measurements as zero or estimated values.

- [ ] **Step 5: Run all relevant tests and manifest validation**

Run: `/opt/conda/bin/python -m pytest -q tests/test_energy.py tests/test_benchmark_nyu_vanilla_gpu_worker.py tests/test_summarize_spn_cross_platform_energy.py tests/test_summarize_spn_cross_platform.py`

Expected: all tests pass and every referenced evidence file hash matches.
