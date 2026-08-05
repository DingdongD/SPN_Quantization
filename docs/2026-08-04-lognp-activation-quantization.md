# Per-Channel LogNP Activation Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a calibrated per-channel LogNP activation QDQ reference backend,
optional bias/weight compensation, and reproducible W8A4/W4A4 evaluation for
the four official NYU depth-completion models.

**Architecture:** Preserve the existing uniform hardware-aligned path by
default. Add focused LogNP observers and quantizers, select them through
explicit runner configurations, and keep the first implementation as a float
QDQ reference. Compensation is fitted only on calibration data and restored
between configurations.

**Tech Stack:** PyTorch, NumPy, pytest, existing NYU runner/export pipeline,
and Matplotlib.

---

## File Map

- Create `scripts/lognp_quantization.py`: transform, inverse, channel observer,
  quantizer, statistics, and compensation helpers.
- Modify `scripts/hardware_aligned_quantization.py`: optional LogNP observer and
  quantizer maps, with uniform mode unchanged by default.
- Modify `scripts/run_nyu_rtn_quantization.py`: `lognp` backend, configs,
  arguments, append-safe tables, manifests, and prediction exports.
- Create `tests/test_lognp_quantization.py`: primitive, calibration, numerical
  safety, and compensation tests.
- Modify `tests/test_hardware_aligned_quantization.py` and
  `tests/test_run_nyu_rtn_quantization.py` for regression and runner coverage.
- Create `scripts/plot_lognp_quantization.py` and
  `tests/test_plot_lognp_quantization.py` for aggregation and plots.
- Output `profile_logs/nyu_lognp_quantization/` for all experiment artifacts.

## Task 1: Transform and Quantizer Primitives

**Files:**
- Create: `scripts/lognp_quantization.py`
- Create: `tests/test_lognp_quantization.py`

- [ ] **Step 1: Write failing tests.** Test round-trip identity for
  `lognp_inverse(lognp_transform(x, alpha), alpha)`, signed A4 code limits
  `[-7, 7]`, unsigned ReLU A4 limits `[0, 15]`, zero preservation, and
  rejection of zero/NaN alpha.

- [ ] **Step 2: Run the focused tests.** Run
  `python -m pytest -q tests/test_lognp_quantization.py` and verify collection
  fails because the new module is absent.

- [ ] **Step 3: Implement exact transform APIs.** Add
  `lognp_transform(tensor, alpha, max_z=24.0)` and
  `lognp_inverse(transformed, alpha, max_z=24.0)`. Validate positive finite
  alpha, broadcast it over channel dimensions, clamp magnitude to max_z, and
  use `torch.expm1(log(2) * abs(z))` in the inverse.

- [ ] **Step 4: Implement `LogNPActivationQuantizer`.** Constructor:
  `LogNPActivationQuantizer(bits, alpha, scale, unsigned=False, max_z=24.0)`.
  `quantize_with_codes(tensor)` must transform, scale, round, clamp to signed
  `[-qmax,qmax]` or unsigned `[0,qmax]`, inverse-transform, and return the
  reconstruction plus int32 codes. Store bits, limits, alpha, scale,
  unsigned, and max_z as detached metadata.

- [ ] **Step 5: Run edge-case tests and commit.** Test values through `1e20`,
  monotonicity, finite output, and all code limits. Run `git diff --check`,
  then commit with `git add scripts/lognp_quantization.py
  tests/test_lognp_quantization.py && git commit -m "feat: add LogNP activation quantizer"`.

## Task 2: Channel Calibration and Error Statistics

**Files:**
- Modify: `scripts/lognp_quantization.py`
- Modify: `tests/test_lognp_quantization.py`

- [ ] **Step 1: Write failing observer tests.** Feed `[N,C,H,W]` inputs with
  different per-channel magnitudes. Assert alpha and scale shape `[C]`, channel
  permutation equivariance, deterministic repeated updates, and no use of
  evaluation tensors.

- [ ] **Step 2: Implement `ChannelLogNPObserver`.** Interface:
  `__init__(sample_limit=4096, epsilon=1e-8)`, `update(tensor)`,
  `freeze(bits, unsigned, alpha_factor=1.0, max_z=24.0)`, `observed`,
  `quantizer()`, and `manifest()`. Keep deterministic CPU float32 samples per
  channel, flattening spatial/leading dimensions. At freeze compute
  `base=max(p50, p99*0.01, epsilon)`, `alpha=base*alpha_factor`,
  `zmax=log2(1+max_abs/alpha)`, and `scale=zmax/qmax`; use alpha/scale 1.0 for
  zero channels.

- [ ] **Step 3: Implement statistics.** Report original-domain MAE, RMSE,
  p50, p75, p99, p99.9, transformed-domain SQNR, zero-code rate, clipping rate,
  sign-flip rate, and non-finite count.

- [ ] **Step 4: Run calibration tests and commit.** Run
  `python -m pytest -q tests/test_lognp_quantization.py tests/test_outlier_mitigation_quantization.py`
  and commit the observer/statistics changes.

## Task 3: Instrumentor Integration

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `tests/test_lognp_quantization.py`

- [ ] **Step 1: Write a failing toy-model integration test.** Use the existing
  Conv-BN-ReLU-Conv model with
  `observe(activation_mode="lognp")`, calibration, `freeze()`, and
  `configure(8, 4, {"encoder"}, activation_mode="lognp")`. Assert Conv/Linear
  and ReLU quantizers are LogNP, ReLU is unsigned, manifest rows contain
  channel alpha/scale, and output is finite. Also assert the default uniform
  path still uses existing quantizers and `sx*sw[o]` bias scales.

- [ ] **Step 2: Add separate state maps.** Keep `observers`, `quantizers`, and
  uniform stats unchanged. Add `lognp_observers`, `lognp_quantizers`, and
  `lognp_stats`; update both observer families in one calibration pass.

- [ ] **Step 3: Add explicit mode arguments.** Add
  `observe(activation_mode="uniform")` and
  `configure(..., activation_mode="uniform", alpha_factor=1.0, max_z=24.0)`.
  LogNP mode applies channel quantizers to Conv/Linear inputs and outputs and
  call-indexed ReLU outputs. Do not use transformed scale for INT32 bias; keep
  original bias and record `bias_contract="reference_float_reconstruction"`.

- [ ] **Step 4: Update hooks and restoration.** Select active maps by mode,
  return reconstructed original-domain values, update matching stats, restore
  folded parameters in `disable()`, and clear LogNP state.

- [ ] **Step 5: Run regressions and commit.** Run
  `python -m pytest -q tests/test_hardware_aligned_quantization.py tests/test_lognp_quantization.py`
  and commit the integration.

## Task 4: Bias and Weight Compensation

**Files:**
- Modify: `scripts/lognp_quantization.py`
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_lognp_quantization.py`

- [ ] **Step 1: Write failing compensation tests.** On a known Linear layer,
  assert bias correction reduces mean residual and least-squares correction
  returns finite `[out_features,in_features]` weights with lower calibration
  MSE.

- [ ] **Step 2: Capture bounded calibration matrices.** For selected modules,
  retain at most 8192 rows. For Conv2d use `torch.nn.functional.unfold` with
  the module's kernel/stride/padding/dilation and sample matching output
  positions; for Linear flatten leading dimensions. Store CPU float32 only for
  requested compensation sites.

- [ ] **Step 3: Implement bias correction.** Add
  `fit_bias_correction(target, reconstructed, bias)` returning
  `bias + (target - reconstructed).mean(dim=0)`. Preserve original bias,
  record correction norms, and apply only after LogNP calibration.

- [ ] **Step 4: Implement regularized least squares.** Add
  `fit_weight_correction(inputs, target, ridge=1e-4)`, computing
  `gram=X.T@X`, `rhs=Y.T@X`, and solving
  `(gram+ridge*I)W.T=rhs.T`. Use reconstructed activation rows as X and
  original module output rows as Y. For W4, pass the fitted result through the
  existing per-output-channel symmetric quantizer, recording weight error and
  outlier percentiles. Reject non-finite results or calibration MSE increases
  above 1 percent.

- [ ] **Step 5: Verify isolation and commit.** Every configuration restores
  original folded parameters before fitting; compensation cannot affect FP32
  or later configurations. Run focused tests and commit.

## Task 5: Runner and Metadata

**Files:**
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] **Step 1: Write configuration tests.** Assert
  `build_lognp_configurations(groups)` returns exactly `FP32`,
  `LOGNP_W8A4_tensor`, `LOGNP_W8A4_channel`,
  `LOGNP_W8A4_channel_bias`, `LOGNP_W8A4_channel_weight`,
  `LOGNP_W4A4_channel_weight`, and `W4A8_full`; only LogNP entries carry
  `activation_mode="lognp"`.

- [ ] **Step 2: Add CLI/backend.** Extend `--quant-backend` with `lognp` and
  add `--lognp-alpha-factor` default `1.0`, `--lognp-max-z` default `24.0`,
  and `--lognp-compensation-samples` default `8192`. Reuse current folding,
  merge/state adapters, sample indices, evaluation indices, and NPZ payloads.

- [ ] **Step 3: Calibrate and configure.** Use the same 128 calibration samples
  as current aligned results. Select LogNP mode only for LogNP configurations;
  preserve existing backend configuration behavior.

- [ ] **Step 4: Persist artifacts.** Write `lognp_calibration.csv`,
  `lognp_manifest.csv`, and `lognp_compensation.csv` with module/kind/group,
  bits, signedness, channel count, alpha/scale percentiles, max_z,
  clipping/zero/sign-flip/nonfinite rates, and compensation method. Metadata
  must state `quantization_execution="float_qdq_reference"` and make no
  integer latency claim.

- [ ] **Step 5: Preserve append semantics.** Replace only selected LogNP rows
  and prediction files under `--append`; preserve unrelated RTN, hardware,
  outlier, and mixed rows. Run runner tests and commit.

## Task 6: Analysis and Visualization

**Files:**
- Create: `scripts/plot_lognp_quantization.py`
- Modify: `scripts/plot_quantized_prediction_comparison.py`
- Create: `tests/test_plot_lognp_quantization.py`

- [ ] **Step 1: Write aggregation tests.** Verify model/config order, FP32
  deltas, p50/p75/p99.9 errors, clipping/zero-code rates, and retained
  non-finite counts on synthetic rows.

- [ ] **Step 2: Implement `aggregate_lognp_rows(rows)` and
  `compare_lognp_configs(rows, baseline="LOGNP_W8A4_tensor")`.** Group by
  model/config/module/kind, calculate weighted metrics, and write
  `lognp_summary.csv` and `lognp_layer_summary.csv`.

- [ ] **Step 3: Implement plots.** Per-model figures contain end-to-end
  RMSE/MAE, activation p99.9 error, clipping/zero-code rates, and compensation
  deltas. Use Arial if installed, no title, horizontal grid behind bars,
  unrotated capitalized x labels, and explicit legends. Reuse the existing
  GT/FP32/quantized/error visualizer for the same 64 samples.

- [ ] **Step 4: Run plot tests and commit.** Run the focused plot tests and
  commit reporting changes.

## Task 7: Four-Model Experiment

**Files:**
- Output: `profile_logs/nyu_lognp_quantization/`

- [ ] **Step 1: Run one-sample preflight** with the existing official converged
  CSPN, DySPN, NLSPN, and CompletionFormer checkpoints. Require finite outputs,
  matching shapes, valid metadata, and unchanged FP32 output after disable.

- [ ] **Step 2: Run all configurations** with the same 128 calibration and 64
  evaluation indices as current aligned results: FP32, all LogNP configs, and
  W4A8. Export prediction payloads for each selected configuration.

- [ ] **Step 3: Validate artifacts.** Require 64 unique NPZ files per
  model/config, identical GT arrays, complete manifests, finite calibration
  parameters, and explicit non-finite counts. Run both Python environments'
  test suites.

- [ ] **Step 4: Generate and inspect outputs.** Run aggregation and prediction
  plotting; inspect four model figures and the combined 64-sample grid for
  white masks, invalid pixels, clipped tails, and label overlap.

- [ ] **Step 5: Write `profile_logs/nyu_lognp_quantization/lognp_findings.md`**
  with per-model uniform A4, tensor/channel LogNP, compensation deltas, p99.9
  activation error, clipping/zero rates, and the float-reference qualification.

## Task 8: Final Verification

- [ ] Run `git diff --check` and `python -m py_compile` on all new/modified
  scripts.
- [ ] Run focused and full tests in the base and `completionformer-py37`
  environments.
- [ ] Verify no existing profile rows or prediction directories outside
  `profile_logs/nyu_lognp_quantization/` were deleted or rewritten.
- [ ] Confirm metadata says `float_qdq_reference` until a LUT/piecewise
  implementation is separately measured.
- [ ] Stage only source, tests, and report files belonging to this feature.
