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
- The official DySPN builder is instantiated and strictly checkpoint-loaded.
  NLSPN and CompletionFormer import a mandatory `DCN` extension that is absent
  in this environment, so their tests verify exact official class declarations
  and selected checkpoint module-key trees without an import or backend
  fallback.

## Fix Round 1

### Exact Fixes

- Preserved every `(module_name, semantic_role)` pair from
  `ModelSemanticAdapter.module_manifest()` in
  `QuantizationModelContract.module_roles`.
- Derived protected modules from retained semantic roles plus each model's
  propagation root. Generic block construction now excludes protected semantic
  modules before blocks are emitted, and the immutable contract rejects any
  protected semantic module in generic weights.
- Added `guidance_logits` to the DySPN, NLSPN, and CompletionFormer protected
  role declarations. Removed protected guidance and confidence-only roots from
  generic tail search topology.
- Required `protected_roles` to be non-empty and unique. Also reject duplicate
  semantic module-role names and duplicate protected modules.
- Added direct integration coverage. DySPN instantiates the official `Model`,
  strictly loads the selected converged checkpoint, validates its full module
  tree, and builds a protected contract. NLSPN and CompletionFormer verify
  their official class declarations and complete selected checkpoint module-key
  trees; their mandatory `DCN` extension is unavailable in this environment,
  so they are not imported or substituted with a fallback.

### Commands And Outputs

```text
PYTHONPATH=. pytest -q tests/test_model_quantization_contracts.py tests/test_official_model_quantization_contracts.py
Result before implementation: 6 failed, 6 passed in 5.49s.

PYTHONPATH=. pytest -q tests/test_model_quantization_contracts.py tests/test_official_model_quantization_contracts.py
Result after implementation: 12 passed in 5.76s.

PYTHONPATH=. pytest -q tests/test_model_quantization_contracts.py tests/test_model_semantic_adapters.py tests/test_propagation_aware_adapters.py tests/test_official_model_quantization_contracts.py
Result: 37 passed in 6.23s.

git diff --check
Result: exit 0 with no output.
```

### Commits

- `aa6e5983883b77262c69da0041ec9d2bd58248a6` `fix: enforce protected semantic model roles`
