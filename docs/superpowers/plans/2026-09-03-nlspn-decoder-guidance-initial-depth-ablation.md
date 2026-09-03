# NLSPN Decoder, Guidance, And Initial-Depth FP16 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a strict 12-candidate NLSPN FP16 ablation that localizes residual error across the shared decoder, guidance decoder, and initial-depth decoder.

**Architecture:** Reuse the corrected NLSPN activation-protection evaluator and explicit IEEE FP16 QDQ. Build all candidates from the measured `EARLY_W16A16` assignment, promote only declared module weights and inputs, and report isolated and cumulative RMSE attribution in separate artifacts.

**Tech Stack:** Python 3.7/3.11, PyTorch, official NLSPN/DCN, pytest, CSV/JSON.

---

### Task 1: Define The Candidate Matrix

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] Add a failing test importing `DECODER_ATTRIBUTION_GROUPS` and `_decoder_attribution_candidates`, then assert the exact six groups and 12 unique candidate names from the design.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py -k decoder_attribution_candidate` and confirm the import failure.
- [ ] Add the six immutable module groups and construct the baseline by selecting `EARLY_W16A16` from `_early_fp16_candidates`.
- [ ] Promote both weight and input activation formats to `fp16_ieee` for each isolated or cumulative module set without changing the source assignment.
- [ ] Run the focused candidate test and confirm all exact formats and source immutability.

### Task 2: Define Isolated And Cumulative Attribution

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] Add failing tests for `_decoder_isolated_attribution_rows` and `_decoder_cumulative_attribution_rows` using synthetic RMSE values.
- [ ] Assert positive isolated recovery as baseline minus isolated RMSE, cumulative marginal recovery as previous prefix minus current prefix, total recovery, shared-decoder interaction, and downstream interaction.
- [ ] Run the focused attribution tests and confirm missing-symbol failures.
- [ ] Implement the two strict attribution helpers with exact candidate coverage checks and no inferred defaults.
- [ ] Run the focused tests and the complete activation-protection test module.

### Task 3: Add The Strict Evaluation Entry Point

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] Add a failing schema test that requires the new CLI experiment name and immutable artifact set.
- [ ] Add `run_decoder_attribution()` using `_reference_activation_protection`, `_evaluate_activation_protection_candidate`, `_assignment_payload`, and the existing paired signal/state diagnostics.
- [ ] Verify every selected module has exactly one effective-weight row with the candidate format and a nonzero changed-element count.
- [ ] Write `summary.csv`, `sample_metrics.csv`, `module_diagnostics.csv`, `effective_weight_metrics.csv`, `propagation_signal_metrics.csv`, `propagation_state_metrics.csv`, `isolated_attribution.csv`, `cumulative_attribution.csv`, and `manifest.json`.
- [ ] Register `decoder-attribution` in the CLI and rerun the complete focused suite.

### Task 4: Run Official CUDA Evaluation And Report Results

**Files:**
- Create: `docs/results/2026-09-03-nlspn-decoder-guidance-initial-depth-ablation-results.md`
- Create runtime output: `profile_logs/nyu_nlspn_decoder_guidance_initial_ablation_64_v1/`

- [ ] Run `py_compile`, Python 3.11 tests, official Python 3.7 tests, and `git diff --check`.
- [ ] Verify CUDA device 1 and import the official DCN extension in the configured Python 3.7 environment.
- [ ] Run the 12 candidates with the fixed 128/64 protocol and no existing-output overwrite.
- [ ] Validate 64 sample rows per configuration, paired reproducibility, finite positive predictions, 18 propagation states, exact sample identities, expected effective formats, and JSON without non-finite values.
- [ ] Document isolated recovery, cumulative marginal recovery, interactions, final FP32 gap, average W/A bits, and the next error boundary supported by the measurements.
