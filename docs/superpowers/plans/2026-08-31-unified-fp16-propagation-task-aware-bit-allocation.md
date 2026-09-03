# Unified FP16 Propagation and Task-Aware Mixed-Precision Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Unify CSPN, DySPN, NLSPN, and CompletionFormer under one strict NYU quantization protocol: keep each model's complete propagation semantic subgraph in FP16, and allocate W/A bits from {4, 6, 8} over the remaining ordinary modules using task-gradient sensitivity and independent weight and activation budgets.

**Architecture:** Existing model contracts identify module ownership, existing propagation adapters own the FP16 boundary, the existing hardware-aligned QDQ path quantizes ordinary Conv/ConvTranspose/Linear modules, and existing sensitivity/allocation modules select bits. A four-model runner composes these interfaces and emits a self-describing manifest plus measured 64-sample results.

**Tech Stack:** Python, PyTorch, existing NYU runtime/checkpoints, existing CUDA QDQ kernels, pytest, CSV/JSON artifacts.

**Spec:** docs/superpowers/specs/2026-08-31-unified-fp16-propagation-task-aware-bit-allocation-design.md

## Global Constraints

- Use the official converged CSPN, DySPN, NLSPN, and CompletionFormer checkpoints and the existing 128-sample stratified calibration set.
- Use the existing fixed 64-sample evaluation identity for every model and method.
- Propagation-owned modules and tensors never enter W4/W6/W8 integer allocation.
- Ordinary module weight and activation budgets are independent and checked exactly from module cost totals.
- Unknown ownership, missing gradients, incomplete sensitivity levels, overlapping ownership, missing calibration data, non-finite outputs, non-positive depth outputs, and budget violations raise immediately.
- Do not add fallback model paths, silent exception recovery, implicit default budgets, or a second quantizer implementation.
- Preserve historical artifacts and label them as legacy protocol results; write new results under a new run root.

---

## Task 1: Lock the propagation FP16 contract

**Files:**
- Modify: spn_quant/propagation/controller.py
- Modify: spn_quant/propagation/adapters.py
- Modify: spn_quant/model_contracts.py
- Test: tests/test_propagation_fp16_contract.py

- [ ] Add failing tests that configure every propagation adapter in explicit fp16 mode and assert that the controller mode is float, the state dtype is torch.float16, and propagation signals are not sent through integer QDQ.
- [ ] Add strict ownership declarations for CSPN, DySPN, NLSPN, and CompletionFormer covering propagation heads, projection/deformable propagation inputs, affinity/offset/confidence/state, normalization, update, and sparse-depth anchor injection.
- [ ] Reject an ownership declaration with an unknown module, missing required propagation site, or overlap with an ordinary allocation site.
- [ ] Implement one explicit shared FP16 configuration path using the existing PropagationQuantController.configure_float("fp16") and adapters; keep the existing integer PA path unchanged for historical comparisons.
- [ ] Ensure the FP16 boundary is applied before any ordinary activation QDQ and that state, coefficient normalization, anchor injection, and every propagation iteration remain in the declared FP16 path.
- [ ] Run pytest -q tests/test_propagation_fp16_contract.py tests/test_propagation_aware_adapters.py.

## Task 2: Exclude propagation from ordinary W/A allocation

**Files:**
- Modify: spn_quant/mixed_precision.py
- Modify: spn_quant/model_contracts.py
- Modify: scripts/run_nyu_model_p3t3_search.py
- Test: tests/test_propagation_allocation_exclusion.py

- [ ] Add failing tests for each model proving that the ordinary allocation registry contains only non-propagation Conv2d, ConvTranspose2d, and Linear sites.
- [ ] Add a registry validation function that checks the disjoint union of ordinary sites and propagation-owned sites and verifies that every ordinary site has both weight and activation costs.
- [ ] Make the evaluator reject an assignment containing a propagation-owned module instead of silently ignoring it.
- [ ] Preserve declared branch-aware concat calibration for ordinary encoder/decoder branches while keeping the propagation FP16 boundary unchanged.
- [ ] Run pytest -q tests/test_propagation_allocation_exclusion.py tests/test_run_nyu_model_p3t3_search.py.

## Task 3: Extend task-gradient sensitivity to independent W/A candidates

**Files:**
- Modify: spn_quant/task_sensitivity.py
- Modify: spn_quant/task_aware_allocation.py
- Test: tests/test_task_sensitivity.py
- Test: tests/test_task_aware_allocation.py

- [ ] Add failing tests for complete per-module score tables at bits {4, 6, 8} and for rejection of missing module/bit entries, non-finite scores, and mismatched costs.
- [ ] Implement strict weight scores sum(abs(g_w * (w - fake_quant(w, b)))) for each ordinary module and bit level.
- [ ] Implement strict activation scores sum(abs(g_x * (x - fake_quant(x, b)))) for each ordinary activation site and bit level using the existing calibrated hardware-aligned activation quantizer.
- [ ] Add explicit weighted-cost accounting for weight and activation element counts and separate average-bit budget validation.
- [ ] Implement deterministic promotion from W4/A4 through W6 and W8, choosing the largest sensitivity reduction per additional storage cost for the selected independent budget; preserve deterministic name ordering for ties.
- [ ] Return serializable score tables, selected assignments, budget totals, and marginal gains without hiding infeasible inputs.
- [ ] Run pytest -q tests/test_task_sensitivity.py tests/test_task_aware_allocation.py.

## Task 4: Build the unified four-model evaluator

**Files:**
- Add: scripts/run_nyu_unified_fp16_task_aware_allocation.py
- Add: configs/four_model_unified_fp16_task_aware.json
- Modify: scripts/nyu_model_runtime.py
- Modify: scripts/evaluate_nyu_selected_quantization.py
- Test: tests/test_run_nyu_unified_fp16_task_aware_allocation.py

- [ ] Add failing CLI/config tests requiring explicit model list, checkpoint identity, calibration metadata, 64 evaluation indices, propagation dtype fp16, W/A bit levels, and separate budgets.
- [ ] Implement the runner by composing NYUModelRuntime, build_model_quantization_contract, HardwareAlignedInstrumentor, propagation adapters, and task sensitivity/allocation primitives already in the repository.
- [ ] Run one FP32 reference, one uniform W8A8 ordinary-module baseline with FP16 propagation, and the selected task-aware mixed assignment for each model.
- [ ] Materialize the exact module assignment and propagation ownership manifest before evaluation; re-evaluate from a fresh official model instance so calibration and selection state cannot leak into measurement.
- [ ] Emit per-model CSV/JSON rows containing pooled metrics, relative FP loss, average W/A bits, ordinary-module counts by bit, propagation dtype, validity counts, and calibration/evaluation identities.
- [ ] Emit a Pareto CSV over the requested independent W/A budgets and mark invalid candidates explicitly as failed records with their strict error message.
- [ ] Run pytest -q tests/test_run_nyu_unified_fp16_task_aware_allocation.py.

## Task 5: Run strict CUDA evaluation and audit outputs

**Files:**
- Generate under: profile_logs/nyu_four_model_unified_fp16_task_aware_64/
- Add: docs/results/2026-08-31-unified-fp16-task-aware-results.md

- [ ] Verify the selected conda environment exposes CUDA, the official model imports, the CUDA QDQ extension, and all four checkpoint identities before launching the run.
- [ ] Run calibration and task-gradient capture on the existing 128 stratified samples without changing preprocessing.
- [ ] Run the fixed 64-sample CUDA evaluation for CSPN, DySPN, NLSPN, and CompletionFormer with FP16 propagation and the selected independent W/A budgets.
- [ ] Audit every manifest for propagation exclusion, exact cost accounting, finite predictions, positive valid depth outputs, anchor validity, and deterministic assignment hashes.
- [ ] Compare FP32, uniform W8A8+FP16 propagation, and task-aware mixed precision using pooled RMSE, mean-sample RMSE, MAE, AbsRel, iRMSE, and invalid-pixel counts.
- [ ] Record which modules receive W4, W6, and W8 for weights and activations, and report whether each model reaches an acceptable loss target without quantizing propagation.
- [ ] Run python scripts/run_nyu_unified_fp16_task_aware_allocation.py --config configs/four_model_unified_fp16_task_aware.json and retain the complete command log beside the run artifacts.

## Task 6: Final repository verification

- [ ] Run the focused unit-test set from Tasks 1-4 and git diff --check.
- [ ] Inspect generated manifests and result tables for missing fields, unintended legacy artifact references, and non-ASCII or temporary-file churn.
- [ ] Summarize the four-model results, the selected W/A allocations, the propagation FP16 boundary, and remaining accuracy or runtime gaps in docs/results/2026-08-31-unified-fp16-task-aware-results.md.
- [ ] Do not claim completion until the CUDA command, artifact audit, and focused tests have all passed.
