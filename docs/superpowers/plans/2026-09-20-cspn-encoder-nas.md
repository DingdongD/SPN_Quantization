# CSPN Encoder NAS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and start a reproducible encoder-only NAS experiment that finds A100-fast CSPN candidates and proves FP32 RMSE non-inferiority to the ResNet18 baseline within 2%.

**Architecture:** Add a self-contained `spn_quant.nas` package for candidate specifications, enumeration, Pareto selection, weight transfer, benchmarking, and statistics. Add a CSPN candidate model that reuses the released decoder and propagation classes while adapting searched encoder channels back to the fixed decoder interfaces. Extend the existing converged trainer only at its model/data/run-name boundaries, then drive preparation and ranking from a dedicated CLI.

**Tech Stack:** Python 3.11, PyTorch 2.7, CUDA events, NumPy, existing NYU HDF5 loader, unittest/pytest.

---

### Task 1: Candidate Specification and Enumeration

**Files:**
- Create: `spn_quant/nas/__init__.py`
- Create: `spn_quant/nas/spec.py`
- Create: `tests/test_cspn_nas_spec.py`

- [ ] **Step 1: Write failing specification tests**

Test that `EncoderSpec` validates aligned, non-decreasing widths; serializes to a stable ID; includes the R18 control; and enumerates projection-only stage modes without duplicates.

```python
from spn_quant.nas.spec import EncoderSpec, enumerate_encoder_specs


def test_r18_spec_has_stable_identity():
    spec = EncoderSpec.r18()
    assert spec.depths == (2, 2, 2, 2)
    assert spec.widths == (64, 128, 256, 512)
    assert spec.slug == "s64-w64-128-256-512-d2-2-2-2"


def test_projection_mode_is_canonical_depth_zero():
    spec = EncoderSpec(stem_width=32, widths=(32, 64, 128, 256),
                       depths=(1, 1, 1, 0))
    assert spec.depths[-1] == 0


def test_invalid_widths_are_rejected():
    with pytest.raises(ValueError, match="non-decreasing"):
        EncoderSpec(32, (32, 64, 48, 128), (1, 1, 1, 1))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_spec.py`
Expected: collection fails because `spn_quant.nas.spec` does not exist.

- [ ] **Step 3: Implement immutable specification and deterministic enumeration**

Use a frozen dataclass with tuple fields, `to_dict`, `from_dict`, `slug`, `r18`, and an iterator over stem widths `(32, 48, 64)`, per-stage multipliers `(0.5, 0.75, 1.0)`, stage-1 depths `(1, 2)`, and later depths `(0, 1, 2)`. Round widths to multiples of 16, enforce non-decreasing stages, and de-duplicate by serialized tuple.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest -q tests/test_cspn_nas_spec.py`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/nas tests/test_cspn_nas_spec.py
git commit -m "feat: define CSPN encoder NAS search space"
```

### Task 2: Searchable CSPN Model

**Files:**
- Create: `models/cspn_encoder_nas.py`
- Create: `tests/test_cspn_nas_model.py`

- [ ] **Step 1: Write failing model-contract tests**

Construct the R18 control and a reduced candidate with one CSPN iteration. Assert output shape `(1, 1, 228, 304)`, encoder feature shapes at output strides 2/4/8/32, decoder adapter channels `(64, 64, 128, 512)`, finite forward output, and fewer parameters for the reduced candidate.

```python
def test_reduced_candidate_preserves_decoder_contract():
    model = build_cspn_nas(EncoderSpec(32, (32, 64, 128, 256),
                                       (1, 1, 1, 0)), cspn_step=1)
    output = model(torch.rand(1, 4, 228, 304))
    assert output.shape == (1, 1, 228, 304)
    assert torch.isfinite(output).all()
    assert model.decoder_channels == (64, 64, 128, 512)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_model.py`
Expected: import fails because `models/cspn_encoder_nas.py` does not exist.

- [ ] **Step 3: Implement candidate stages and adapters**

Reuse `BasicBlock`, `Gudi_UpProj_Block`, `Gudi_UpProj_Block_Cat`, the two output heads, and `Affinity_Propagate`. Implement `ProjectionStage` as stride-2 1x1 Conv-BN-ReLU. Implement an encoder whose stage output adapters map stem/stage1/stage2/stage4 to decoder channels 64/64/128/512. Copy the released decoder forward path unchanged after those adapters.

- [ ] **Step 4: Run model tests and full regression suite**

Run: `python -m pytest -q tests/test_cspn_nas_model.py`
Expected: pass.

Run: `python -m pytest -q`
Expected: 357 existing tests plus new tests pass.

- [ ] **Step 5: Commit**

```bash
git add models/cspn_encoder_nas.py tests/test_cspn_nas_model.py
git commit -m "feat: add searchable CSPN encoder model"
```

### Task 3: Deterministic Weight Transfer

**Files:**
- Create: `spn_quant/nas/weights.py`
- Create: `tests/test_cspn_nas_weights.py`

- [ ] **Step 1: Write failing prefix-transfer tests**

Fill source tensors with known values and verify 4D Conv, 1D BatchNorm, and scalar counters copy the overlapping prefix. Assert shape-incompatible tensors remain initialized and are reported, and a repeated transfer gives the same report and state.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_weights.py`
Expected: import fails because the transfer helper does not exist.

- [ ] **Step 3: Implement transfer report and prefix slicing**

Implement `transfer_prefix_state(target, source_state)` returning copied, partial, skipped, and missing key lists. Copy the intersection slice along every dimension only when tensor ranks match; require exact dtype; never mutate the source dictionary.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_cspn_nas_weights.py`
Expected: pass.

```bash
git add spn_quant/nas/weights.py tests/test_cspn_nas_weights.py
git commit -m "feat: transfer R18 weights into NAS candidates"
```

### Task 4: Split Manifest, Pareto Ranking, and Run Identity

**Files:**
- Create: `spn_quant/nas/search.py`
- Create: `tests/test_cspn_nas_search.py`

- [ ] **Step 1: Write failing search-policy tests**

Test a 6,700-index deterministic split produces 6,030/670 disjoint sets, stores a source-list SHA256, and reproduces exactly for one seed. Test Pareto ranking minimizes RMSE/parameters/latency, uses crowding distance for ties, and retains `max(8, ceil(N/4))` rows.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_search.py`
Expected: import fails because search helpers do not exist.

- [ ] **Step 3: Implement manifest and NSGA-II selection helpers**

Implement pure functions `make_search_split`, `validate_search_split`, `pareto_rank`, `crowding_distance`, and `select_successive_halving`. Reject duplicate/missing indices and non-finite objective values with explicit errors.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_cspn_nas_search.py`
Expected: pass.

```bash
git add spn_quant/nas/search.py tests/test_cspn_nas_search.py
git commit -m "feat: add deterministic CSPN NAS selection policy"
```

### Task 5: A100 Benchmarking

**Files:**
- Create: `spn_quant/nas/benchmark.py`
- Create: `tests/test_cspn_nas_benchmark.py`

- [ ] **Step 1: Write failing benchmark tests**

Use a fake timing backend to verify warm-up count, 1,000 measured iterations, five repetitions, median-of-medians, global p95, non-finite rejection, output-shape rejection, and busy-GPU preflight rejection.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_benchmark.py`
Expected: import fails because benchmark helpers do not exist.

- [ ] **Step 3: Implement benchmark protocol**

Provide `benchmark_model(model, input_tensor, warmup=200, iterations=1000, repeats=5)` using CUDA Events and synchronization, plus `gpu_environment()` parsing `nvidia-smi` CSV. Disable TF32 inside a context manager and restore previous flags on exit. Refuse measurement when another compute PID occupies the selected GPU.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_cspn_nas_benchmark.py`
Expected: pass without requiring CUDA through the fake backend.

```bash
git add spn_quant/nas/benchmark.py tests/test_cspn_nas_benchmark.py
git commit -m "feat: benchmark CSPN NAS candidates on A100"
```

### Task 6: Trainer Integration

**Files:**
- Modify: `scripts/train_nyu_iteration_sweep.py`
- Create: `tests/test_cspn_nas_trainer.py`

- [ ] **Step 1: Write failing trainer-boundary tests**

Test `build_cspn` loads a JSON candidate spec, returns NAS metadata, transfers an optional control checkpoint, and leaves the legacy `resnet18` path unchanged. Test loader index manifests apply exact train/dev subsets and `--run-name` controls output directory naming.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_trainer.py`
Expected: parser/build helpers reject the new arguments.

- [ ] **Step 3: Add minimal NAS arguments and boundary hooks**

Add `--cspn-encoder-spec`, `--cspn-control-checkpoint`, `--split-manifest`, and `--run-name`. Build the candidate only when a spec path is supplied. Apply train/dev indices after dataset construction. Include hashes and transfer report in `meta.json`; include the serialized spec in checkpoint identity.

- [ ] **Step 4: Verify focused and full tests**

Run: `python -m pytest -q tests/test_cspn_nas_trainer.py tests/test_run_nyu_rtn_quantization.py`
Expected: pass.

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_nyu_iteration_sweep.py tests/test_cspn_nas_trainer.py
git commit -m "feat: train CSPN NAS candidates with existing runner"
```

### Task 7: Experiment CLI and Stage-Zero Launch

**Files:**
- Create: `scripts/run_cspn_encoder_nas.py`
- Create: `tests/test_run_cspn_encoder_nas.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Test `prepare` writes immutable split/spec manifests, counts full and encoder parameters, records failed candidates, resumes only exact identities, and emits deterministic training commands. Test `rank` reads rung results and writes selected candidate IDs.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_run_cspn_encoder_nas.py`
Expected: import fails because the CLI does not exist.

- [ ] **Step 3: Implement prepare, benchmark, and rank subcommands**

`prepare` generates `split.json`, candidate JSON files, and `candidates.csv`. `benchmark` loads each candidate and appends A100 results atomically. `rank` applies the tested Pareto policy. Every output records source revision, dataset-list hash, seed, PyTorch/CUDA versions, and failure reason.

- [ ] **Step 4: Document exact commands**

Add README commands for CPU preparation/smoke, A100 benchmark, 5/15/40-epoch training, and ranking. State that fake-QDQ timing is not INT8 latency.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest -q tests/test_run_cspn_encoder_nas.py`
Expected: pass.

```bash
git add scripts/run_cspn_encoder_nas.py tests/test_run_cspn_encoder_nas.py README.md
git commit -m "feat: orchestrate CSPN encoder NAS experiment"
```

- [ ] **Step 6: Start stage zero**

Run:

```bash
SPN_DATA_ROOT=/workspace/CSPN/cspn_pytorch \
python scripts/run_cspn_encoder_nas.py prepare \
  --train-list /workspace/CSPN/cspn_pytorch/datalist/nyudepth_hdf5_train.csv \
  --control-checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --output-root output/cspn_encoder_nas_20260920 \
  --seed 20260920
```

Expected: split/spec manifests and candidate cost table exist; CPU model smoke checks pass. If all GPUs remain busy, benchmark exits with a recorded preflight failure and does not emit latency claims.

### Task 8: Paired Non-Inferiority Statistics

**Files:**
- Create: `spn_quant/nas/statistics.py`
- Create: `scripts/report_cspn_encoder_nas.py`
- Create: `tests/test_cspn_nas_statistics.py`

- [ ] **Step 1: Write failing hierarchical-bootstrap tests**

Use synthetic seed-by-sample errors to verify deterministic 10,000-replicate output, paired resampling, one-sided 95% upper bound, a passing candidate below the 2% margin, and a failing candidate above it.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_statistics.py`
Expected: import fails because statistics helpers do not exist.

- [ ] **Step 3: Implement and report**

Implement `paired_hierarchical_bootstrap` over matched seed/sample arrays. The report CLI validates identical sample IDs and seeds, emits JSON/CSV/Markdown, and reports unrounded delta, margin, upper confidence bound, RMSE, MAE, ABS_REL, and DELTA1.25.

- [ ] **Step 4: Verify full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add spn_quant/nas/statistics.py scripts/report_cspn_encoder_nas.py tests/test_cspn_nas_statistics.py
git commit -m "feat: report CSPN NAS non-inferiority"
```

### Task 9: W8A8 Candidate Verification

**Files:**
- Modify: `scripts/export_nyu_predictions.py`
- Modify: `spn_quant/adapters/cspn.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`
- Create: `tests/test_cspn_nas_w8a8.py`

- [ ] **Step 1: Write failing candidate-load and semantic-site tests**

Create a reduced candidate run directory with `args.json`, `meta.json`, its
encoder spec, and a checkpoint. Assert `export_nyu_predictions.build_model`
reconstructs the NAS candidate exactly. Install the CSPN semantic adapter and
assert searched encoder Conv outputs, fixed decoder outputs, heads, and
propagation signals are observed. Assert `W8A8_full` includes candidate
adapters and projection-only stages.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_cspn_nas_w8a8.py`
Expected: candidate reconstruction or required encoder-role validation fails.

- [ ] **Step 3: Extend model provenance and CSPN module-role matching**

Allow the local CSPN NAS model source in provenance checks, reconstruct it
through the saved trainer arguments, and extend the encoder activation rule to
cover named `skip_adapters`, `bottleneck_adapter`, and `ProjectionStage`
modules. Keep all existing official CSPN role names unchanged.

- [ ] **Step 4: Verify focused W8A8 tests**

Run:

```bash
python -m pytest -q \
  tests/test_cspn_nas_w8a8.py \
  tests/test_run_nyu_rtn_quantization.py \
  tests/test_model_semantic_adapters.py
```

Expected: pass.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add scripts/export_nyu_predictions.py spn_quant/adapters/cspn.py \
  tests/test_run_nyu_rtn_quantization.py tests/test_cspn_nas_w8a8.py
git commit -m "feat: verify W8A8 on CSPN NAS candidates"
```
