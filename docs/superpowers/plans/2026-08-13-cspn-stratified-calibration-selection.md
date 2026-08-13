# CSPN Stratified Calibration Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and audit a deterministic train-only 128-sample NYU calibration set that covers raw input and selected official CSPN FP32 activation distributions better than uniform random sampling.

**Architecture:** A pure NumPy/PyTorch selection module owns descriptor formulas, robust normalization, grouped distance, tail cover, k-center, k-medoids, and audit metrics. A dedicated CSPN runner owns official dataset parity, FP32 activation hooks, sample-role isolation, artifact serialization, and the real 6,700-sample workflow. Existing quantization runners remain unchanged and consume the generated index artifact only when explicitly configured later.

**Tech Stack:** Python, NumPy, PyTorch, torchvision, CUDA, pytest.

---

### Task 1: Raw descriptors and robust feature space

**Files:**
- Create: `spn_quant/calibration_selection.py`
- Create: `tests/test_calibration_selection.py`

- [ ] **Step 1: Write failing raw descriptor tests**

Use synthetic RGB, depth, and up to 500 sparse coordinates. Assert depth
mean/p50/p95/max/valid ratio, luminance mean/std, RGB contrast, fixed-threshold
Sobel edge density, quadrant occupancies, 16x16 grid occupancy, centroid spread,
and rejection of empty sparse depth or a sparse count above 500.

- [ ] **Step 2: Run the descriptor tests and verify RED**

Run:

```bash
python -m pytest tests/test_calibration_selection.py -q
```

Expected: import failure because `spn_quant.calibration_selection` does not
exist.

- [ ] **Step 3: Implement descriptor records and formulas**

Add immutable feature schema constants and a `raw_descriptor` function. Inputs
must be finite CHW tensors with matching spatial dimensions. Use direct errors
for invalid rank, range, valid-depth count, and sparse count. Do not repair or
drop samples.

- [ ] **Step 4: Write failing robust-normalization and distance tests**

Assert exact median/IQR, rejection of zero IQR, frozen transforms, equal group
weighting, finite values, and diagnostic feature exclusion.

- [ ] **Step 5: Implement normalizer and grouped distance**

Return explicit schema, median, IQR, transformed matrix, and grouped squared
distance. Do not use sklearn preprocessing or inferred feature defaults.

- [ ] **Step 6: Run focused tests and commit**

```bash
python -m pytest tests/test_calibration_selection.py -q
git add spn_quant/calibration_selection.py tests/test_calibration_selection.py
git commit -m "feat: add calibration descriptor feature space"
```

### Task 2: Deterministic stratified selection algorithms

**Files:**
- Modify: `spn_quant/calibration_selection.py`
- Modify: `tests/test_calibration_selection.py`

- [ ] **Step 1: Write failing tail-cover tests**

Construct known low/high p5/p95 conditions. Assert deterministic greedy set
cover, multi-tail sample selection, aggregate-tail-distance fill, exact budget,
index tie-breaking, complete condition coverage, and failure when a budget
cannot cover all conditions.

- [ ] **Step 2: Implement strict tail cover**

Return selected positions, condition membership, selection order, and reason.
Every non-diagnostic dimension has low and high conditions. Never relax p5/p95
or fill randomly.

- [ ] **Step 3: Write failing k-center and k-medoids tests**

Assert farthest-first k-center on a known matrix, exact medoid membership,
deterministic ties, convergence, exact count, and errors for duplicate indices,
invalid distance matrices, or empty clusters.

- [ ] **Step 4: Implement deterministic coverage algorithms**

Implement greedy k-center over a supplied pairwise distance matrix. Implement
k-medoids with farthest-first initialization, nearest-medoid assignment, and
minimum within-cluster distance updates until stable. Keep the 1,024 by 1,024
distance matrix float32.

- [ ] **Step 5: Write and implement disjoint split tests**

Verify current random-128, audit-512, eligible-6,188, 16 random baselines, and
all pairwise disjointness requirements. Audit selection must exclude the
current random-128 before seeded sampling.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/test_calibration_selection.py -q
git add spn_quant/calibration_selection.py tests/test_calibration_selection.py
git commit -m "feat: add deterministic stratified calibration selection"
```

### Task 3: CSPN dataset and activation descriptor runner

**Files:**
- Create: `scripts/run_nyu_cspn_stratified_calibration.py`
- Create: `tests/test_run_nyu_cspn_stratified_calibration.py`

- [ ] **Step 1: Write failing runner contract tests**

Assert exact required CLI fields and counts: calibration 128, audit 512,
candidate 1,024, candidate tail 256, final tail 32, random baseline count 16,
expected activation owner declarations, explicit seeds, CUDA device, and no
evaluation-list argument.

- [ ] **Step 2: Implement deterministic raw dataset access**

Use the existing NYU HDF5 loader to produce pre-normalization augmented RGB,
depth, and exact sparse depth. Build the CSPN RGB through the existing
`legacy_cspn_rgb` conversion. Validate fixed-index depth and CSPN RGB parity
against `CspnOfficialDataset` before scanning. A mismatch is fatal.

- [ ] **Step 3: Write failing activation-hook tests**

Use a small module graph with reused ReLUs. Assert exact call-index owner
capture, one tensor per declared owner, p99/max/ratio/channel-imbalance values,
reset per forward, and direct failure on missing or repeated captures.

- [ ] **Step 4: Implement strict FP32 activation collector**

Load the official CSPN checkpoint through the existing loader. Keep the model
in FP32 eval mode with all quantization adapters disabled. Capture the six
declared owners and profile each unique index across candidates, audit, current
baseline, and 16 random baselines exactly once.

- [ ] **Step 5: Implement stage-one and stage-two orchestration**

Reserve audit indices first, scan 6,188 raw rows, choose 256 raw-tail plus 768
k-center candidates, profile candidate activations, choose 32 combined-tail
plus 96 k-medoids final samples, then profile the remaining audit/baseline
union. Preserve ordered index identity and explicit sample roles.

- [ ] **Step 6: Run runner tests and commit**

```bash
python -m pytest tests/test_run_nyu_cspn_stratified_calibration.py \
  tests/test_calibration_selection.py -q
git add scripts/run_nyu_cspn_stratified_calibration.py \
  tests/test_run_nyu_cspn_stratified_calibration.py
git commit -m "feat: select CSPN calibration samples from train descriptors"
```

### Task 4: Train-only audit and artifact contract

**Files:**
- Modify: `spn_quant/calibration_selection.py`
- Modify: `scripts/run_nyu_cspn_stratified_calibration.py`
- Modify: `tests/test_calibration_selection.py`
- Modify: `tests/test_run_nyu_cspn_stratified_calibration.py`

- [ ] **Step 1: Write failing coverage metric tests**

Assert calibration range coverage, audit p01/p50/p99/max, finite quantile
ratios, standardized one-dimensional Wasserstein distance, nearest-calibration
p50/p95, activation max exceedance, and comparison against 16 random sets.

- [ ] **Step 2: Implement coverage metrics and acceptance**

Compute every metric from the frozen stage-two normalizer. Acceptance requires
complete tails, stratified p95 nearest distance below random mean, activation
uncovered-max count no greater than current random-128, finite values, exact
counts, and disjoint sets.

- [ ] **Step 3: Write failing serialization tests**

Assert exact output filenames, required JSON keys, CSV schemas, sample counts,
selection reasons, checkpoint provenance, and absence of image files. Missing
rows or duplicate identities must fail before metadata is written.

- [ ] **Step 4: Implement explicit CSV/JSON/Markdown output**

Write calibration and audit indices, raw and activation descriptors, candidate
and final selection rows, descriptor coverage, distance coverage, activation
range coverage, metadata, and a concise text report. No plot path is added.

- [ ] **Step 5: Run focused regression and commit**

```bash
python -m pytest tests/test_calibration_selection.py \
  tests/test_run_nyu_cspn_stratified_calibration.py -q
git add spn_quant/calibration_selection.py \
  scripts/run_nyu_cspn_stratified_calibration.py \
  tests/test_calibration_selection.py \
  tests/test_run_nyu_cspn_stratified_calibration.py
git commit -m "feat: audit CSPN calibration distribution coverage"
```

### Task 5: Real CUDA selection, audit, and report

**Files:**
- Create: `profile_logs/nyu_cspn_stratified_calibration_128/`
- Create: `docs/2026-08-13-cspn-stratified-calibration-results.md`

- [ ] **Step 1: Run official CSPN selection**

Use the converged `cspn_iter24/best.pt`, train list from its `args.json`, data
root `/workspace/CSPN/cspn_pytorch`, baseline seed `20260812`, audit seed
`20260813`, 128/512/1,024 counts, 256/32 tail budgets, 16 explicit random
seeds, and an available A100.

- [ ] **Step 2: Audit artifacts before interpretation**

Verify 128 final indices, 512 audit indices, 1,024 candidates, 6,188 eligible
raw rows, expected activation rows for every profiled index and six owners,
complete tail conditions, finite descriptors, no index leakage, checkpoint
identity, and no image artifacts.

- [ ] **Step 3: Compare coverage and write results**

Report raw and activation range coverage, p99/max ratios, nearest-distance
p50/p95, uncovered activation owners, tail composition, and comparison with
the current and 16 random baselines. State acceptance or failure directly.

- [ ] **Step 4: Run full verification**

```bash
python -m pytest -q
git diff --check -- . ':(exclude)tests/test_qdrop_reconstruction.py'
```

- [ ] **Step 5: Commit the result report**

```bash
git add docs/2026-08-13-cspn-stratified-calibration-results.md
git commit -m "docs: report CSPN stratified calibration coverage"
```
