# Task 8 Report: Fix Round 1/5

Date: 2026-08-26

Fix base: clean committed Task 8 review state
`7cbabaa8ae1943157facaf9cbb2d0e1f1d524c79` in
`/workspace/SPN_Quantization/.worktrees/cspn-lsqplus-hawq`.

Binding inputs read before implementation:

- `task-8-review.md`
- `task-8-brief.md`
- `docs/superpowers/specs/2026-08-25-three-model-selected-quantization-adaptation-design.md`
- `docs/superpowers/plans/2026-08-25-three-model-selected-quantization-adaptation.md`

No subagent was dispatched. No formal search, reconstruction, multi-epoch QAT,
or fixed-64 evaluation was launched before review.

## Result

All round-1 review findings are implemented:

1. Every official runtime activates the exact indexed CUDA device with
   `torch.cuda.set_device(index)` and asserts `torch.cuda.current_device()`
   before extension import, checkpoint loading, or official model construction.
2. A committed official full-model smoke harness covers the exact eight-entry
   primitive/method matrix for DySPN, NLSPN, and CompletionFormer.
3. Static inputs have exact semantic schemas and cross-file validation.
4. The DAG now generates and validates all five static inputs per model from
   the official train/validation splits before method artifacts.
5. The final plan has 70 explicit jobs and preserves all P3/T3, HAWQ, artifact,
   evaluation, aggregation, and publication dependencies.
6. README and inventory documentation describe the executable workflow and
   explicit no-fallback policy.

## CUDA Device Root Fix

`scripts/nyu_model_runtime.py` now provides
`activate_explicit_cuda_device(device, family)`. It requires `cuda:<index>`,
checks availability and device count, calls `torch.cuda.set_device(index)`, and
asserts the resulting current device. `NYUModelRuntime.build_model()` invokes
it before `_assert_required_cuda_extension()` and official construction.

The direct QDrop reconstruction process invokes the same helper before its
official student/teacher models are built. The smoke extension primitive also
asserts the exact current device. No command or environment sets
`CUDA_VISIBLE_DEVICES`, and no alternate device or interpreter is selected.

After this fix, the former NLSPN failure reproduced successfully on `cuda:1`:
current device `1`, prediction shape `[1,1,228,304]`, 18 propagation states,
and no illegal access. The final NLSPN and CompletionFormer full hard smokes
then completed on `cuda:1` and `cuda:2` respectively. Therefore this report
makes no native-kernel limitation claim.

## Static Inputs And DAG

New production files:

- `spn_quant/nyu_static_inputs.py`
- `scripts/prepare_nyu_three_model_static_inputs.py`

Each model's `prepare_static_inputs` job uses its absolute configured Python,
full replacement environment, and exact indexed CUDA device. It declares five
outputs:

- `calibration_metadata.json`
- `calibration_indices.json`
- `evaluation_protocol.json`
- `weight_cost_rows.csv`
- `activation_cost_rows.csv`

The producer samples 256 seeded train candidates, retains 32 distribution-tail
samples, selects 96 weighted k-medoids, writes the ordered 128 calibration
identity, writes the configured ordered fixed-64 validation protocol, and
captures weight MACs and activation traffic from an official forward.

The following `validate_static_inputs` job verifies exact JSON/CSV schemas,
model/dataset/checkpoint/data-root identities, checkpoint and dataset-list
hashes, split identities and bounds, ordered 128/64 identities, evaluation
seed, descriptor schema, unique positive costs, complete ordered cost coverage,
and all cross-file artifact hashes. P3/T3, HAWQ trace, and QAT depend on this
receipt; selected PTQ and mixed QAT additionally depend on P3/T3.

Fresh plan command:

```bash
env PYTHONHASHSEED=0 \
  PYTHONPATH=/workspace/SPN_Quantization/.worktrees/cspn-lsqplus-hawq \
  /opt/conda/bin/python \
  scripts/launch_nyu_three_model_quantization.py plan \
  --config /workspace/SPN_Quantization/.worktrees/cspn-lsqplus-hawq/configs/three_model_selected_quantization.json \
  --launch-spec /workspace/SPN_Quantization/.worktrees/cspn-lsqplus-hawq/configs/three_model_quantization_launch.json
```

Result: exit `0`, 70 job manifests, three static producers, three semantic
validators, 30 exact model/method evaluations, and no CVD entry. Plan:

`/workspace/SPN_Quantization/profile_logs/nyu_three_model_selected_quantization/launch/launch_plan.json`

SHA256:
`1057ebddbc36daacda18abde6665cabc6c064a93821982c95ce75bafde853da3`.

The plan persists each command, full environment, indexed device, inputs and
revisions, outputs, dependencies, log, start/end placeholders, and exit-status
placeholder. Execution refreshes revisions and persists running/completed/
failed timestamps and status.

## Official Smoke Harness

`scripts/smoke_nyu_selected_quantization.py` executes, in exact order:

1. FP32
2. RTN W8A8
3. RTN W4A4
4. QDrop W6A6 hard deployment
5. BRECQ W6A6 hard deployment
6. one LSQ++ W4A4 optimizer step
7. one HAWQ probe
8. one P3/T3 candidate

Every record requires native CUDA execution, official propagation execution,
finite `[1,1,228,304]` output, and model-specific state, affinity, contraction,
and anchor policy invariants. QDrop/BRECQ require materialized hard weights;
LSQ++ requires a finite nonzero gradient and one optimizer step.

The official DySPN `grid_sample` forward lacks autograd double backward. This
was reproduced only after the current-device fix. The HAWQ mode is now an exact
model contract, not a caught-error fallback: DySPN uses central block finite
differences with epsilon `0.001`; NLSPN and CompletionFormer use autograd block
HVP. Trace artifact format 3 persists this mode and the loader rejects changes.
DySPN's one probe executed 48 complete official forwards and 1,440 official
`grid_sample` calls.

### Exact CUDA Evidence

| Model | Python/device | UTC start/end | Result | Manifest SHA256 |
| --- | --- | --- | --- | --- |
| DySPN | `/opt/conda/bin/python`, `cuda:0` | `04:16:53.840580Z` / `04:17:28.634896Z` | exit 0 | `1d031d0b24e071edea9da0517d88ce2cddce835cb62909abdf23d52100928425` |
| NLSPN | Python 3.7, `cuda:1` | `04:17:51.033382Z` / `04:18:34.840525Z` | exit 0 | `bd461725ddfcf64b69291923cbe3c716669147e360d8f2507f88df562100875c` |
| CompletionFormer | Python 3.7, `cuda:2` | `04:18:00.276904Z` / `04:20:01.483592Z` | exit 0 | `f4ea6cff7d69f275af9dcba3f3a5117e14dde97957574674ebdcda928309b21e` |

Manifest paths:

- `/workspace/SPN_Quantization/profile_logs/task8_fix_round1_smokes/dyspn-v4/official_one_sample_smoke.json`
- `/workspace/SPN_Quantization/profile_logs/task8_fix_round1_smokes/nlspn-v6/official_one_sample_smoke.json`
- `/workspace/SPN_Quantization/profile_logs/task8_fix_round1_smokes/completionformer-v3/official_one_sample_smoke.json`

All 24 method records have finite expected output and valid propagation. Native
and official-call evidence includes:

| Model/method | Native calls | Official propagation calls | Additional evidence |
| --- | ---: | ---: | --- |
| DySPN FP32/RTN/P3 | 1 each | 30 each | real torchvision deform-conv primitive plus official grid propagation |
| DySPN QDrop/BRECQ | 1 each | 1,590 each | hard weights materialized |
| DySPN LSQ++ | 1 | 90 | loss `0.8515208959579468`, grad norm `0.9554214025167344` |
| DySPN HAWQ | 1 | 1,440 | 24 blocks, finite-difference HVP |
| NLSPN FP32/RTN/LSQ/HAWQ/P3 | 26 each | 26 each | LSQ grad norm `0.5473163645449525`, HAWQ 26 blocks |
| NLSPN QDrop/BRECQ | 1,482 each | 1,482 each | hard weights materialized |
| CompletionFormer FP32/RTN/LSQ/HAWQ/P3 | 26 each | 26 each | LSQ grad norm `0.7403242621192769`, HAWQ 38 blocks |
| CompletionFormer QDrop/BRECQ | 2,106 each | 2,106 each | hard weights materialized |

## TDD RED/GREEN Evidence

Production changes followed failing tests before edits. Principal cycles:

1. CUDA activation RED:
   `PYTHONPATH=. pytest -q tests/test_nyu_model_runtime.py tests/test_qdrop_reconstruction_runner.py`
   reported three failures for the missing activation helper/call. GREEN:
   `6 passed`.
2. Static contracts RED: `PYTHONPATH=. pytest -q tests/test_nyu_static_inputs.py`
   failed collection because `spn_quant.nyu_static_inputs` was absent. GREEN:
   `5 passed`.
3. Static DAG RED: launcher tests failed because producer/validator jobs and
   dependencies were absent. GREEN: exact cardinality 70 and required edges.
4. Fold controls RED: selected PTQ and QAT parser tests rejected the missing
   explicit fold arguments. GREEN: selected/QDrop `28 passed`; QAT/parser
   `47 passed`.
5. Propagation policy RED: four tests failed when NLSPN/CompletionFormer were
   evaluated with DySPN anchor requirements. GREEN: model-specific preserve
   policy in all P3/QDrop/QAT/evaluation/smoke paths.
6. Python 3.7 hook RED: the official PyTorch 1.10 hook API rejected
   `prepend=True`. GREEN: exact legacy hook test plus both Python interpreters.
7. CompletionFormer reconstruction RED: tests exposed non-finite deep-block
   gradients. GREEN: forward loss remains exact while its derivative is
   normalized; non-finite gradients are rejected before optimizer steps.
8. DySPN QAT RED: stochastic-depth blocks attempted to mutate parametrized
   non-leaf weights. GREEN: official deterministic stochastic-depth branch is
   retained while quantizer parameters train.
9. DySPN HAWQ RED:
   `PYTHONPATH=. pytest -q tests/test_hawq_trace.py tests/test_run_nyu_model_hawq_trace.py`
   failed collection for the absent finite-difference API. Artifact RED then
   failed `assert 2 == 3`. GREEN: `31 passed` across HAWQ and smoke tests.

## Final Verification

Affected suite:

```text
PYTHONPATH=. /opt/conda/bin/python -m pytest -q <18 affected test files>
266 passed, 2 skipped in 24.56s
```

Repository-wide Python 3.11 suite:

```text
PYTHONPATH=. /opt/conda/bin/python -m pytest -q
1524 passed, 2 skipped, 2 failed, 1 warning, 16 subtests passed in 290.47s
```

The two failures are explicit Python-version assertions in the official NLSPN
and CompletionFormer builder tests (`(3,11) != (3,7)`). Exact rerun under the
declared replacement environment:

```text
env <declared CompletionFormer environment> \
  /opt/conda/envs/completionformer-py37/bin/python -m pytest -q \
  tests/test_official_model_quantization_contracts.py::test_official_nlspn_builder_strictly_loads_checkpoint_and_contract \
  tests/test_official_model_quantization_contracts.py::test_official_completionformer_builder_strictly_loads_checkpoint_and_contract
2 passed in 9.84s
```

Both Python 3.11 and Python 3.7 compiled every affected production module.
`git diff --check` passed. Production scans found no broad catches, `/tmp`,
`tempfile`, CVD assignment, `cuda:3`, automatic Python/device selection, or
new dictionary default access.

## Formal Status

Formal execution was intentionally not started. The fresh plan and all 70
per-job manifests are ready for review. The explicit next command is the
documented `execute` operation using the exact plan path above; it will first
generate and semantically validate all 15 per-model static files before any
P3/T3, HAWQ, PTQ, or QAT artifact job can run.

Commit subject: `fix: close task 8 launch review findings`.
