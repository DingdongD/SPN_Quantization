# Task 1 Report: Strict Model Quantization Contracts

## Status

Completed. The selected DySPN, NLSPN, and CompletionFormer models now resolve
immutable generic-quantization contracts with strict ownership validation.

## Files Changed

- `spn_quant/model_contracts.py`: immutable contract records, strict
  validation, and model-contract builder.
- `spn_quant/adapters/base.py`: class-level semantic module manifest.
- `spn_quant/adapters/dyspn.py`: protected roles and prefix/tail patterns.
- `spn_quant/adapters/nlspn.py`: protected roles and prefix/tail patterns.
- `spn_quant/adapters/completionformer.py`: protected roles plus complete
  four-stage transformer prefix and tail patterns.
- `tests/test_model_quantization_contracts.py`: synthetic official-shaped
  model trees and strict validation coverage.

## Design Decisions

- Reused `resolve_qdrop_targets()` as the existing explicit, fail-closed
  module-pattern resolver. Contract blocks expand its exact block roots into
  Conv2d, ConvTranspose2d, and Linear module names.
- Added `ModelSemanticAdapter.module_manifest()` so contracts consume the
  adapter's model-specific semantic module view without installing hooks or
  changing model execution.
- Each adapter owns protected propagation roles and explicit prefix/tail search
  patterns. CompletionFormer additionally exposes Q/K/V and concat input edges
  from the existing joint-QDrop ownership records.
- `QuantizationModelContract` rejects empty blocks, duplicate weight ownership,
  duplicate activation owners, protected roles in generic activation owners,
  protected modules in generic weights, invalid search groups, and unknown
  attention/concat edges.
- `protected_modules` is required explicitly; there is no fallback value.

## Tests

Red phase:

```text
pytest -q tests/test_model_quantization_contracts.py
Result: collection error, No module named spn_quant.

python -m pytest -q tests/test_model_quantization_contracts.py
Result: collection error, No module named spn_quant.model_contracts.
```

The first command used a launcher that did not add the worktree root to
`sys.path`. The second command reached the intended missing-contract import.

Green and regression verification:

```text
python -m pytest -q tests/test_model_quantization_contracts.py
Result: 7 passed in 2.86s.

python -m pytest -q tests/test_model_quantization_contracts.py tests/test_model_semantic_adapters.py tests/test_propagation_aware_adapters.py
Result: 31 passed in 3.07s.

PYTHONPATH=. pytest -q tests/test_model_quantization_contracts.py tests/test_model_semantic_adapters.py tests/test_propagation_aware_adapters.py
Result: 31 passed in 3.10s.

git diff --check
Result: exit 0 with no output.
```

## Commit Hashes

- `80c03bc1c9a21ee6e310471f4a0135f1f0099230` `feat: add strict model quantization contracts`

## Concerns

- The bare `pytest` executable in this environment lacks the repository root
  on `sys.path`; use `python -m pytest` or `PYTHONPATH=. pytest` for this
  worktree.
- No official CUDA-backed model was instantiated for this unit-level task.
  The tests use full-shaped synthetic module trees, while Task 2 will validate
  contracts against the official runtime facade and checkpointed models.
