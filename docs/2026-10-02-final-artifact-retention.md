# Final artifact retention

This file records the retained evidence after the October 2 NAS experiment
cleanup. Deleted directories contained failed, incomplete, smoke, screening,
or superseded outputs and can be regenerated from the tracked scripts.

## Four-model accuracy and compression

- `profile_logs/structured_deep_channel_nas_quant_fullval_v1/`
- `profile_logs/structured_combined_depth_nas_quant_fullval_v1/`
- `profile_logs/structured_hidden_channel_nas_quant_fullval_v1/`
- `profile_logs/nyu_four_model_int_mixed_precision_1pct/`
- `docs/2026-10-02-four-model-structured-pruning-lowbit-validation.md`
- `docs/2026-10-02-hardware-aware-spn-nas-strategy.md`

The first two directories contain the selected 654-image validation summaries
and parameter-compression manifests. The hidden-channel directory retains the
safer NLSPN Pareto point. The mixed-precision directory retains source QAT
checkpoints needed to reproduce the selected assignments.

## Vanilla reference checkpoints

The unmodified software baselines remain under
`/root/demo/artifacts/source128_reference_ckpts/`, together with their original
arguments, metrics, run summaries, and source-training evidence.

| Model | Checkpoint size | SHA256 |
| --- | ---: | --- |
| CSPN | 104,116,325 bytes | `623dd30bf71ab029849b4988e3fd59a4ed9acd51d296e572524ad584c896a3a5` |
| DySPN | 65,750,241 bytes | `6e4d691d8efcb71476e9ff72cb90dd647e95b8d3891cdea4cc9d4052830567dc` |
| NLSPN | 64,604,637 bytes | `8559a22d1115d68a21ba1a97280220dd3c4b63f52e1f3c6205157c74b8d3d8c9` |
| CompletionFormer | 334,428,835 bytes | `6a2a3ae5468881a39e21d8d9ff02aaab7fd54772d9a1588207172dd8d7c8ba33` |

The official model source trees and runtime adapters were also retained. These
checkpoints remain the immutable references for every vanilla-relative RMSE
gate; NAS or quantized checkpoints must not replace them in baseline rows.

## CSPN NAS

- `output/cspn_encoder_nas_20260920/structured_decoder_width320/`
- `output/cspn_encoder_nas_20260920/structured_decoder_fullval_v1/`
- `output/cspn_encoder_nas_20260920/official_validation/`
- `output/cspn_encoder_nas_20260920/training/final-candidate-seed*/`
- `output/cspn_encoder_nas_20260920/training/final-control-seed*/`
- `output/cspn_encoder_nas_20260920/u250/lowbit/mixed_w4_physical_regions_v184/`
- `output/cspn_encoder_nas_20260920/u250/execution/agentflow_cmodel/`

## U250 host artifacts

- `/root/demo/artifacts/`
- `/root/demo/validation_runs_20261002/u250-completionformer-board-repair6-full-python39/iteration-01/`
- `/root/demo/validation_runs_20261002/rhb-cspn-full-chain-board-python39/`
- `/root/demo/validation_runs_20261002/rhb-completionformer-replay-cafcfab/`

Reusable toolchains remain in `/root/demo/u250-toolchain-native` and related
environment directories. Per-run toolchain snapshots are not retained.

## Cleanup result

The cleanup removed approximately 66 GiB of obsolete experiment outputs and
additional rebuildable caches. Root-filesystem free space increased from
9.2 GiB to more than 74 GiB before the final cache cleanup. No selected
full-validation summary, final checkpoint, retained U250 package, source file,
or compiler environment was removed.
