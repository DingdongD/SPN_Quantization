# CSPN Scale-Aware Group-8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic RMS-ranked consumer-side Group-8 activation quantization for official CSPN and compare it with the current contiguous MinMax W4A4 baseline on real NYU data.

**Architecture:** A focused scale-aware module owns RMS grouping, permutations, paired Conv weight transformation, and a permutation-aware activation-quantizer wrapper. The existing hardware instrumentor accepts an explicit permutation contract, applies the weight transformation before W4, and maps diagnostics back to original channel order. A dedicated CSPN runner reuses the audited 128-sample calibration and fixed 64-sample evaluation pipeline.

**Tech Stack:** Python, PyTorch, CUDA, NumPy, Matplotlib, pytest.

---

### Task 1: RMS grouping and permutation primitives

**Files:**
- Create: `spn_quant/scale_aware_grouping.py`
- Create: `tests/test_scale_aware_grouping.py`
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [x] **Step 1: Write failing tests for channel RMS accumulation**

Add tests proving `ChannelMinMaxObserver.channel_rms()` accumulates squared
values across updates, preserves channel order, and rejects use before
observation. Use two NCHW updates with analytically known RMS values.

- [x] **Step 2: Run the RMS tests and verify RED**

Run:

```bash
python -m pytest tests/test_hardware_aligned_quantization.py -q
```

Expected: FAIL because `channel_rms` does not exist.

- [x] **Step 3: Implement exact channel RMS accumulation**

Extend `ChannelMinMaxObserver.update()` with float64 CPU `square_sum` and an
integer scalar count per channel. Implement `channel_rms()` as:

```python
return torch.sqrt(self.square_sum / float(self.scalar_count)).to(torch.float32)
```

Do not estimate RMS from MinMax values and do not silently synthesize an
unobserved result.

- [x] **Step 4: Write failing grouping and permutation tests**

Cover:

- stable ascending `(rms, channel_index)` ordering;
- exact groups of eight;
- tied and zero RMS channels;
- bijective inverse permutation;
- lower summed dispersion than a deliberately mismatched contiguous grouping;
- Conv2d `x[:, P]` plus `W[:, P]` equivalence;
- ConvTranspose2d `x[:, P]` plus `W[P]` equivalence;
- direct rejection of non-finite RMS, duplicate indices, unsupported modules,
  and channel counts not divisible by eight.

- [x] **Step 5: Run the primitive tests and verify RED**

Run:

```bash
python -m pytest tests/test_scale_aware_grouping.py -q
```

Expected: FAIL because `spn_quant.scale_aware_grouping` does not exist.

- [x] **Step 6: Implement minimal grouping primitives**

Implement focused public functions:

```python
def build_scale_aware_grouping(channel_rms, group_size, epsilon):
    ...

def inverse_permutation(permutation):
    ...

def permute_activation(tensor, permutation, channel_dim):
    ...

def permute_input_weight(module, weight, permutation):
    ...

def grouped_channel_maximum(channel_maximum, permutation, group_size):
    ...
```

Return a frozen record containing permutation, inverse, RMS, contiguous and
scale-aware dispersions. Use stable `torch.argsort`; do not add a solver,
random tie breaking, or fallback grouping.

- [x] **Step 7: Verify and commit the primitives**

Run both focused test files, then commit only Task 1 files.

### Task 2: Permutation-aware activation QDQ and instrumentor integration

**Files:**
- Modify: `spn_quant/scale_aware_grouping.py`
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_scale_aware_grouping.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [x] **Step 1: Write failing quantizer-wrapper tests**

Construct a four-channel test with Group-2 scales and a nontrivial permutation.
Prove that `quantize_for_consumer()` returns:

1. execution activation in permuted order;
2. reconstructed activation in original order;
3. integer codes in original order;
4. `scale_for(reference)` in original channel order.

- [x] **Step 2: Implement `PermutedGroupedActivationQuantizer`**

Wrap the existing grouped quantizer through explicit constructor fields. Do not
use attribute forwarding or `getattr`. Expose the uniform QDQ contract fields
and implement:

```python
execution, comparable, comparable_codes = quantizer.quantize_for_consumer(x)
```

The wrapped grouped quantizer receives `x[:, permutation]`; comparable values,
codes, and scales are mapped through the stored inverse permutation.

- [x] **Step 3: Write failing instrumentor contract tests**

Tests must prove:

- only declared Conv input Group-8 specs accept permutations;
- permutation coverage and bijection are strict;
- a declared permutation requires the consumer weight group to be W4;
- Conv2d and ConvTranspose2d weights are permuted before W4;
- disabling QDQ restores original weights;
- W4 scales and weight error match the unpermuted configuration;
- input activation recorder receives original-order reference, reconstruction,
  codes, and scales;
- output and ReLU quantizers remain unchanged.

- [x] **Step 4: Integrate explicit activation permutations**

Add `activation_permutations` to
`configure_components_with_ranges()` and `configure()`. Validate every mapping
against the input `QuantSpec`, observer channel count, group size, module type,
and enabled weight groups. During configuration:

1. reorder the input-channel extent before Group-8 MinMax range construction;
2. wrap the resulting grouped quantizer;
3. reorder the original Conv input weight before W4 QDQ;
4. return the execution tensor from the pre-hook;
5. feed original-order comparable tensors into existing statistics and
   activation recorders.

An empty explicit mapping preserves every existing configuration.

- [x] **Step 5: Run integration regressions and commit**

Run:

```bash
python -m pytest \
  tests/test_scale_aware_grouping.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_cspn_activation_resolution.py -q
```

Commit only Task 2 files after all tests pass.

### Task 3: Dedicated official CSPN experiment runner

**Files:**
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Modify: `tests/test_run_nyu_cspn_activation_resolution.py`
- Create: `scripts/run_nyu_cspn_scale_aware_grouping.py`
- Create: `tests/test_run_nyu_cspn_scale_aware_grouping.py`

- [x] **Step 1: Write failing base-runner plumbing tests**

Extend the CSPN configuration contract with an explicit
`activation_permutations` tuple. Verify ordinary configurations declare an
empty tuple and `_configure_quantized()` passes the exact mapping to the
instrumentor without changing rotation or propagation configuration.

- [x] **Step 2: Implement base-runner plumbing**

Add the field to `_configuration`, `_derived_configuration`, and
`_configure_quantized`. Access it directly with `config["activation_permutations"]`;
do not add a runtime default or fallback.

- [x] **Step 3: Write failing scale-aware runner tests**

Verify the dedicated runner:

- declares exactly `W4A4_G8_MINMAX` and `W4A4_G8_SCALE_AWARE`;
- selects only ordinary Conv/ConvTranspose input owners with divisible Group-8
  channel counts;
- derives every permutation from `channel_rms()` after the existing 128-sample
  calibration pass;
- preserves baseline ranges and all rotation/SPN policies;
- emits strict grouping, sample, activation, block, propagation, and prediction
  coverage;
- requires exactly 128 calibration and 64 evaluation samples.

- [x] **Step 4: Implement the dedicated runner**

Reuse official model loading, folding, strict-site validation,
`base._calibrate`, `base.run_configuration`, fixed evaluation identities, and
prediction payload writing. Write artifacts under:

```text
profile_logs/nyu_cspn_scale_aware_group8/
```

Export `grouping_manifest.csv` with one row per channel and
`grouping_summary.csv` with one row per eligible site. Include calibration
indices, epsilon, eligible-site count, permutations, inverse permutations, and
unchanged propagation/bias/guidance contracts in metadata.

- [x] **Step 5: Run runner contract tests and commit**

Run the new runner tests together with CSPN activation-resolution regressions,
then commit Task 3 files.

### Task 4: Real CUDA evaluation and result analysis

**Files:**
- Create: `scripts/plot_cspn_scale_aware_grouping.py`
- Create: `tests/test_plot_cspn_scale_aware_grouping.py`
- Create: `docs/2026-08-13-cspn-scale-aware-grouping-results.md`

- [x] **Step 1: Run the real experiment**

Use the official converged CSPN checkpoint, the same seed and 128 real NYU
calibration samples as the static-calibration study, the fixed 64 evaluation
indices, MinMax W4A4 Group-8, FP32 bias/guidance, and A8/Q13/INT32 propagation.

Run:

```bash
PYTHONPATH=. python scripts/run_nyu_cspn_scale_aware_grouping.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --sample-metrics /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/cspn/sample_metrics.csv \
  --data-root /workspace/CSPN/cspn_pytorch \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_scale_aware_group8 \
  --device cuda:0 --seed 20260812 --calibration-samples 128 \
  --sample-capacity 256 --group-size 8 --dispersion-epsilon 1e-12 \
  --fold-max-error 0.05
```

- [x] **Step 2: Audit completed artifacts**

Assert:

- `2 * 64` finite sample rows with exact identities;
- exactly two prediction payloads per evaluation identity;
- every eligible input has one bijective permutation and inverse;
- every group has eight channels;
- no output, ReLU, rotation, or SPN owner is permuted;
- paired W4 scales and reconstruction errors are invariant within tolerance;
- activation error decomposition remains numerically closed;
- propagation anchor and contraction constraints remain valid.

- [x] **Step 3: Write plotting tests and generate comparisons**

Reuse the existing CSPN prediction/error visual style. Generate aggregate
metric comparison, per-site dispersion versus SQNR change, per-sample RMSE
delta, and GT/FP32/baseline/scale-aware prediction comparisons. Plot only from
completed CSV/NPZ artifacts.

- [x] **Step 4: Analyze the result**

Report whether lower RMS dispersion reduces A4 zero collapse and whether those
local gains survive the initial-depth head and propagation. Quantify median,
P90, and maximum per-sample RMSE delta; do not select the method from visual
quality or tensor SQNR alone.

Run:

```bash
PYTHONPATH=. python scripts/plot_cspn_scale_aware_grouping.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_scale_aware_group8 \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_scale_aware_group8/figures \
  --expected-samples 64 --dpi 160
```

- [x] **Step 5: Run complete verification and commit**

Run:

```bash
python -m pytest -q
git diff --check -- . ':(exclude)tests/test_qdrop_reconstruction.py'
```

Commit implementation, plotter, tests, and result report while leaving the
unrelated `tests/test_qdrop_reconstruction.py` modification untouched.
