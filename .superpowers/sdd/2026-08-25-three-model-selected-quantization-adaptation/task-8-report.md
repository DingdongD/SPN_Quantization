# Task 8 Report: Multi-GPU Launch and End-to-End Validation

Date: 2026-08-26

Base state: clean committed Task 7 at `b37284c` in
`/workspace/SPN_Quantization/.worktrees/cspn-lsqplus-hawq`.

## Result

Task 8 adds a strict two-phase launcher for the official DySPN, NLSPN, and
CompletionFormer selected-quantization matrix. Planning and execution are
separate operations. No formal search, reconstruction, QAT, or fixed-64
evaluation was launched before review.

The launcher has 64 explicit jobs:

- 21 jobs per model across three independent configured GPU lanes;
- one final cross-model summary job;
- 30 formal evaluation jobs, one for each exact model/method pair.

The immutable formal method order is:

1. `fp32`
2. `rtn_w8a8`
3. `rtn_w4a4`
4. `qdrop_w6a6`
5. `brecq_w6a6`
6. `hawq_mixed_le6`
7. `lsqplus_w6a6`
8. `lsqplus_w4a4`
9. `mixed_task_aware`
10. `p3_t3_mixed_ptq`

## Files

Created:

- `configs/three_model_quantization_launch.json`
- `scripts/launch_nyu_three_model_quantization.py`
- `tests/test_launch_nyu_three_model_quantization.py`
- this report

Modified:

- `README.md`
- `docs/2026-08-20-quantization-framework-inventory.md`
- `spn_quant/propagation/adapters.py`
- `tests/test_propagation_aware_adapters.py`

The DySPN propagation files were changed because real hard-QDQ smoke testing
found that the production P3/T3 validator rejected every DySPN candidate:
DySPN exposed state and affinity statistics but did not expose the exact
confidence-gated sparse-anchor blend. The adapter now records one
`anchor_injection` invariant row per quantized iteration. This is statistics
only and does not change propagation output or model state.

## Explicit Runtime Contract

The launch specification fixes these lanes and never remaps them:

| Model | Python | Device |
| --- | --- | --- |
| DySPN | `/opt/conda/bin/python` | `cuda:0` |
| NLSPN | `/opt/conda/envs/completionformer-py37/bin/python` | `cuda:1` |
| CompletionFormer | `/opt/conda/envs/completionformer-py37/bin/python` | `cuda:2` |

Every subprocess receives the exact replacement environment declared for its
model. `CUDA_VISIBLE_DEVICES` is rejected. Python paths must be absolute,
existing executables. Devices must match `cuda:<integer>`. There is no Python,
GPU, environment, precision, backend, or idle-device fallback.

Conv-BN folding is explicitly disabled with `--skip-conv-bn-fold` on all three
lanes. A real NLSPN smoke run measured a folded FP32 maximum absolute output
error of `0.46452332`, above the configured `0.05` equivalence guard. The
approved runner interface requires an explicit fold choice but does not require
folding.

## DAG And Artifact Gates

Each model lane contains this dependency structure:

```text
p3_t3_mixed_ptq -> selected_ptq
p3_t3_mixed_ptq -> mixed_task_aware
hawq_trace -> hawq_allocation -> hawq_mixed_le6_qat
lsqplus_w4a4_qat
lsqplus_w6a6_qat

selected_ptq + four terminal QAT jobs
  -> formal_artifacts
  -> evaluate_fp32
  -> evaluate_rtn_w8a8
  -> evaluate_rtn_w4a4
  -> evaluate_qdrop_w6a6
  -> evaluate_brecq_w6a6
  -> evaluate_hawq_mixed_le6
  -> evaluate_lsqplus_w6a6
  -> evaluate_lsqplus_w4a4
  -> evaluate_mixed_task_aware
  -> evaluate_p3_t3_mixed_ptq
  -> aggregate
  -> plot
```

All three plot jobs precede `cross_model_summary`. The graph rejects cycles,
missing dependencies, duplicate output owners, self dependencies, and any
DAG-produced input whose producer is not a transitive predecessor.

`formal_artifacts` validates the exact selected PTQ matrix, all five PTQ hard
deployment manifests, all four terminal completed QAT checkpoints, P3/T3 and
HAWQ support artifacts, and hashes every artifact before the first formal
evaluation can run. The cross-model publisher requires the exact 3 by 10
method matrix, fixed sample count 64, and finite nonnegative pooled metrics.

## Provenance

`plan` writes `launch/launch_plan.json` and one JSON manifest per job. Each
manifest contains:

- exact command and full replacement environment;
- configured indexed CUDA device;
- input paths and SHA256/size revisions;
- declared primary and supporting outputs;
- explicit dependencies and log path;
- state, UTC start/end timestamps, and exit status.

Planning hashes each unique existing static input once. Future DAG-produced
inputs remain explicitly pending until their producer completes. Execution
captures all input revisions again immediately before each subprocess, records
running/completed/failed state atomically, and refuses changed plans, changed
completed inputs, missing outputs, existing output collisions, or failed resume
states.

## TDD Evidence

The initial launcher test failed during collection because
`scripts.launch_nyu_three_model_quantization` did not exist. Focused red/green
cycles then covered:

- exact P3/T3, HAWQ, artifact, evaluation, and cross-model dependencies;
- all three models, exact ten-method order, and exact 64-job cardinality;
- configured Python/device use and `CUDA_VISIBLE_DEVICES` rejection;
- exact runner arguments, outputs, and no-fold flag;
- successful and failed job provenance manifests;
- static planned-input revisions;
- generated-input transitive dependency enforcement;
- exact completed 30-row cross-model publication;
- mandatory explicit CLI operation.

The DySPN regression test failed before its implementation with zero
`anchor_injection` rows versus three expected rows, then passed with one exact
zero-error row per propagation iteration. The stricter indexed-device and
generated-input dependency tests both failed before implementation, then all
three targeted tests passed.

## Test Evidence

Final Task 8 and directly affected suite:

```text
PYTHONPATH=. /opt/conda/bin/python -m pytest -q \
  tests/test_model_quantization_contracts.py \
  tests/test_experiment_config.py \
  tests/test_nyu_model_runtime.py \
  tests/test_mixed_precision.py \
  tests/test_run_nyu_selected_ptq.py \
  tests/test_run_nyu_model_hawq_trace.py \
  tests/test_model_method_qat.py \
  tests/test_model_task_loss.py \
  tests/test_train_nyu_selected_qat.py \
  tests/test_evaluate_nyu_selected_quantization.py \
  tests/test_plot_nyu_selected_quantization.py \
  tests/test_launch_nyu_three_model_quantization.py \
  tests/test_propagation_aware_adapters.py

186 passed, 2 skipped in 18.58s
```

Raw repository-wide Python 3.11 run:

```text
PYTHONPATH=. /opt/conda/bin/python -m pytest -q

1498 passed, 2 skipped, 2 failed, 1 warning, 16 subtests passed
in 295.52s
```

The only two failures assert that their interpreter is Python 3.7 before
testing the official NLSPN and CompletionFormer builders. Running exactly those
tests in the declared environment produced:

```text
env -i <declared CompletionFormer environment> \
  /opt/conda/envs/completionformer-py37/bin/python -m pytest -q \
  tests/test_official_model_quantization_contracts.py::test_official_nlspn_builder_strictly_loads_checkpoint_and_contract \
  tests/test_official_model_quantization_contracts.py::test_official_completionformer_builder_strictly_loads_checkpoint_and_contract

2 passed in 9.60s
```

Both `/opt/conda/bin/python` and the declared Python 3.7 interpreter compiled
the launcher and DySPN adapter successfully. `git diff --check` passed. A scan
found no broad exception catches, `/tmp`, `tempfile`, interpreter discovery, or
device-remapping assignment in production Task 8 files.

## CUDA Smoke Evidence

All runs used the declared interpreter, full declared environment, and exact
indexed device. No `CUDA_VISIBLE_DEVICES` remap or `cuda:3` fallback occurred.

Official FP32 one-sample primitives:

| Model | Device | Extension/operator | Shape | Finite | States | Allocated bytes |
| --- | --- | --- | --- | --- | --- | --- |
| DySPN | `cuda:0` | `torchvision.ops.deform_conv2d` / `torchvision::deform_conv2d` | `[1,1,228,304]` | true | 6 | 1030930432 |
| NLSPN | `cuda:1` | `DCN` | `[1,1,228,304]` | true | 18 | 1060674560 |
| CompletionFormer | `cuda:2` | `DCN` | `[1,1,228,304]` | true | 18 | 1562986496 |

DySPN official hard-QDQ one-sample results under the final no-fold policy:

| Candidate | RMSE | Finite | Propagation valid | Reproducible |
| --- | ---: | --- | --- | --- |
| RTN W8A8 | 0.4644116790542781 | true | true | true |
| RTN W4A4 | 0.47107367083862856 | true | true | true |
| first-block P3/T3 | 0.4630156288339363 | true | true | true |

Bounded method primitives ran independently on `cuda:0`, `cuda:1`, and
`cuda:2`. Each lane completed one W6 adaptive-rounding plus A6 reconstruction
and hardening step for QDrop, one deterministic W6/A6 BRECQ-style hardening
step, one LSQ++ W6A6 optimizer step with finite gradients, and one Hutchinson
HAWQ trace probe. All outputs were finite. Exact HAWQ estimates were
`1.5111110210418701` on `cuda:0` and `1.5111545324325562` on both Python 3.7
lanes. LSQ++ losses were `0.10522070527076721` on `cuda:0` and
`0.1052420511841774` on `cuda:1`/`cuda:2`. QDrop losses were
`0.1945728212594986`, `0.19653131067752838`, and `0.19653131067752838` on
`cuda:0`, `cuda:1`, and `cuda:2`, respectively. BRECQ losses in the same order
were `0.0586673878133297`, `0.0586748942732811`, and `0.0586748942732811`.

The official NLSPN and CompletionFormer hard-QDQ evaluator paths could not
complete on their currently assigned devices. `CUDA_LAUNCH_BLOCKING=1`
diagnostics pinned both failures to the required native
`modulated_deformable_im2col_cuda` call:

```text
RuntimeError: CUDA error: an illegal memory access was encountered
```

NLSPN failed in
`NLSPN_ECCV20/src/model/modulated_deform_conv_func.py`; CompletionFormer failed
in `CompletionFormer/src/model/modulated_deform_conv_func.py`. These were
reported rather than hidden by remapping to idle `cuda:3`. The earlier official
FP32 primitives prove that the checkpoints, model classes, sample shapes, and
extension imports were valid, while the blocking traces identify the current
native-kernel limitation for the instrumented hard path.

## Formal Run Status

The exact documented planning command exits 1 at the first missing static
input:

```text
FileNotFoundError: static launch input is missing:
/workspace/SPN_Quantization/profile_logs/nyu_three_model_selected_quantization/dyspn/calibration_metadata.json
```

All 15 declared per-model static inputs are currently absent: calibration
metadata, calibration index contracts, evaluation protocols, weight cost rows,
and activation cost rows for each of the three models. The launcher created no
`launch/` directory after this failure. No multi-hour formal job was started.

The next valid operation is to publish and review those immutable inputs, run
`plan`, inspect all 64 manifests, and only then invoke the separately documented
`execute` command.

## Commit

Planned commit message:

```text
feat: launch three-model quantization evaluation
```
