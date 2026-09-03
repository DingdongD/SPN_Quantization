# FP4/FP8 Layer And Group Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add strict per-layer FP4/FP8 assignment for weights and activations, with independent weighted budgets for encoder, decoder, fusion, attention, and concat.

**Architecture:** Keep the existing quantization model contracts and FP16 propagation boundary. Add a deterministic semantic-group classifier for weight modules and activation owners, then allocate FP8 promotions independently inside each group using existing task-sensitivity scores and cost tables. The evaluator consumes the complete assignment without format fallback; generic boundaries, CompletionFormer Q/K/V, and concat branches retain their existing calibrated scale paths.

**Tech Stack:** Python, PyTorch, pytest, existing NYU unified runtime and contract/cost/sensitivity artifacts.

---

### Task 1: Define group classification and budget allocation

**Files:**
- Modify: `spn_quant/fp_mixed_precision.py`
- Test: `tests/test_fp_mixed_precision.py`

- [ ] Add tests for deterministic classification of encoder, decoder, fusion, attention, and concat owners; reject unknown names rather than assigning a fallback group.
- [ ] Add tests proving weight and activation budgets are checked independently per group and that allocations never exceed the configured weighted average bits.
- [ ] Add a test proving every contract weight and activation owner is classified exactly once.
- [ ] Implement `classify_weight_module`, `classify_activation_owner`, and a grouped allocation function using FP4 as the base and FP8 promotion by descending task sensitivity per group.
- [ ] Return group assignments, weighted fractions, and budget audits in immutable structures suitable for JSON serialization.

### Task 2: Connect grouped assignments to FP evaluators

**Files:**
- Modify: `scripts/run_nyu_four_model_fp4_fp8.py`
- Modify: `spn_quant/adapters/completionformer_joint.py`
- Modify: `scripts/hardware_merge_adapters.py`
- Test: `tests/test_fp_instrumentor.py`

- [ ] Build separate weight and activation sensitivity maps from the existing score tables, preserving the owner-level attention and concat roles.
- [ ] Generate model-specific grouped candidates from configuration budgets and validate complete contract coverage before evaluation.
- [ ] Pass per-module weight formats to the generic instrumentor and per-owner formats to ordinary, attention, and concat execution paths.
- [ ] Preserve independent branch scales and explicitly record concat output format derivation in the manifest.
- [ ] Keep propagation in FP16 and exclude propagation-owned modules from all FP4/FP8 groups.

### Task 3: Add protocol configuration and artifact accounting

**Files:**
- Create: `configs/four_model_fp4_fp8_group_budget.json`
- Modify: `scripts/run_nyu_four_model_fp4_fp8.py`
- Test: `tests/test_run_nyu_four_model_fp4_fp8.py`
- Modify: `docs/results/2026-09-01-four-model-fp4-fp8-results.md`

- [ ] Define independent `weight_average_bits` and `activation_average_bits` limits for each semantic group, with explicit model overrides where a group is absent.
- [ ] Record per-group format fractions, average bits, promoted owners/modules, and feasibility in each manifest and summary row.
- [ ] Keep the existing uniform and global mixed candidates as baselines; add grouped candidates with stable names.
- [ ] Document that budget percentages are cost-weighted, not layer-count percentages.

### Task 4: Verify the implementation

**Files:**
- No source changes unless a test exposes a defect.

- [ ] Run focused FP format, mixed precision, instrumentor, and runner tests.
- [ ] Run the full existing quantization test subset.
- [ ] Run a lightweight four-model contract/assignment validation and inspect all generated group audits.
- [ ] Run `git diff --check` and Python compilation checks.
