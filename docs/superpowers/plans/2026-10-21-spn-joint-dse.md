# SPN Joint Array/Memory DSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible BF16 SPN joint DSE that searches array and memory architectures and reports latency-area-energy Pareto candidates.

**Architecture:** Extend the existing capacity-aware mapping model to support independent activation and weight CACTI macros. Add a separate joint-cost layer so analytical PPA assumptions do not contaminate cycle modeling, then orchestrate explicit architecture profiles through the four full workload traces and validate all generated artifacts.

**Tech Stack:** Python dataclasses, pytest, pinned LLMCompass systolic-cycle primitive, pinned CACTI 7.0, CSV/JSON, matplotlib.

---

### Task 1: Independent SRAM macros and access accounting

**Files:**
- Modify: `spn_quant/hardware/memory_dse.py`
- Modify: `tests/test_memory_dse.py`

- [x] Add failing tests for different activation/weight capacities, widths, latencies, and read/write byte conservation.
- [x] Run the focused tests and verify that the old single-macro contract fails them.
- [x] Extend `MemoryArchitecture`, `MemoryTraffic`, and `MemorySimulationResult` with independent macro properties and access directions.
- [x] Run the focused tests and the existing memory-DSE regression tests.

### Task 2: Analytical area and dynamic-energy model

**Files:**
- Create: `spn_quant/hardware/joint_dse.py`
- Create: `tests/test_joint_dse.py`

- [x] Add failing tests for the pinned LLMCompass FP16 area proxy, CACTI access energy, DRAM/MAC energy, hard constraints, equal-model aggregation, and three-objective domination.
- [x] Run the tests and verify failure due to the missing joint module.
- [x] Implement validated cost assumptions, architecture points, layer/model/candidate aggregation, hard constraints, and deterministic Pareto extraction.
- [x] Run the focused test module.

### Task 3: Joint study runner and artifact validator

**Files:**
- Create: `scripts/run_spn_joint_dse.py`
- Create: `tests/test_run_spn_joint_dse.py`

- [x] Add an end-to-end failing test with small four-model profiles and five architecture classes.
- [x] Verify it fails because the runner is absent.
- [x] Implement configuration loading, parallel simulation, CSV/JSON/PNG output, provenance, and independent validation.
- [x] Verify serial and parallel results are identical and tampered Pareto/capacity data is rejected.

### Task 4: CACTI sweep and default architecture study

**Files:**
- Modify: `scripts/run_cacti_sram.py`
- Modify: `tests/test_cacti_runner.py`
- Create: `configs/cacti/sram_*.json`
- Create: `configs/spn_joint_dse_bf16_lpddr4.json`
- Create: `scripts/run_four_model_joint_dse.sh`

- [x] Add a failing test that multiple macro configurations share one temporary CACTI build.
- [x] Implement multi-config CACTI execution while retaining the one-config CLI.
- [x] Add conservative, balanced, aggressive, activation-heavy, and weight-heavy profiles using 2/4/8 MiB macro choices.
- [x] Add a wrapper that characterizes unique macros once and runs the joint study.

### Task 5: Documentation and verification

**Files:**
- Modify: `reports/README.md`
- Modify: `README.md`

- [x] Document objectives, hard constraints, model equations, confidence, and invocation.
- [x] Run all focused tests, then attempt the complete test suite; record unrelated pre-existing failures explicitly.
- [x] Run a real four-model joint DSE or, if runtime is prohibitive, a validated representative study and report the exact remaining execution command.
- [x] Inspect the diff and generated metadata for accidental scope expansion or unsupported sign-off claims.
