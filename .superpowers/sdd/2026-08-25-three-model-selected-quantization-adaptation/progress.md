# SDD ledger — plan: docs/superpowers/plans/2026-08-25-three-model-selected-quantization-adaptation.md

Spec: docs/superpowers/specs/2026-08-25-three-model-selected-quantization-adaptation-design.md
Base commit: 692492d

## Pre-flight Review

| Scope | Producer / consumer | Finding | Ruling |
| --- | --- | --- | --- |
| Task 1 | Contracts consumed by Tasks 3-7 | Interface is explicit and model-owned | Proceed |
| Task 2 | Runtime consumed by Tasks 3-8 | Runtime must load existing official builders without fallback | Proceed |
| Tasks 3 and 6 | Generic assignments consumed by QAT | CSPN serialized assignments require compatibility aliases | Preserve existing tuple payload shape |
| Tasks 4 and 7 | PTQ artifacts consumed by evaluation | Hard-deployment manifest is the required boundary | Reject fake-quant-only artifacts |
| Tasks 5 and 6 | HAWQ assignment consumed by QAT | HAWQ has separate W/A <=6 budgets | Persist and validate both audits |
| Tasks 6 and 7 | QAT checkpoints consumed by evaluation | Canonical FP32 master state plus method state is required | Reject incomplete checkpoints |
| Tasks 7 and 8 | Evaluator consumed by launcher | Cross-method publication waits for exact artifact matrix | Proceed |
| Task 1 | Tests and implementation | Required fake official module trees can exercise contracts without CUDA | Proceed |
| Task 2 | Tests and implementation | Exact external environments are runtime config, not inferred | Proceed |
| Task 3 | Tests and implementation | P3/T3 labels are model-relative, not fixed layer names | Proceed |
| Task 4 | Tests and implementation | Existing RTN/QDrop code remains authoritative | Extend through explicit contract inputs |
| Task 5 | Tests and implementation | Weight-only Hessian score cannot silently imply activation sensitivity | Keep separate cost budgets and disclose objective components |
| Task 6 | Tests and implementation | Semantic state alignment differs by model | Consume semantic adapter captures only |
| Task 7 | Tests and implementation | Summary names ten configurations including FP32 | Enforce exact ordered set |
| Task 8 | Tests and implementation | GPU selection must remain explicit | No automatic idle-GPU fallback |
| Task 9 | Cleanup | Deletion is destructive and requires validated manifests | Do not execute cleanup without artifact proof |

Ruling: Implementation remains on the existing isolated worktree and feature branch because it is not main/master and contains the prerequisite CSPN LSQ++/HAWQ commits.

Task 1: fix round 1/5 (3 addressed, 2 open — NLSPN/CompletionFormer official contract integration is not exercised; required empty blocks can be silently omitted; commits e2813ea..86ab3e2)
Task 1: fix round 2/5 (2 addressed, 0 open; commits 86ab3e2..8ba415a)
Task 1: complete (strict contracts, protected semantic ownership, and official-model checkpoint integration validated; commits 80c03bc..8ba415a)
Task 2: fix round 1/5 (2 addressed, 0 open; commit ad87dba)
Task 2: complete (strict config, checkpoint identity, official builders, and native extension validation; commits 9c9d83a..ad87dba)
Task 3: fix round 1/5 (2 addressed, 0 open; commit d3058ee)
Task 3: complete (generic allocation registry, executable hard-deployment P3/T3 search, and CSPN compatibility; commits e8345de..d3058ee)
Task 4: fix round 1/5 (3 addressed, 1 new compatibility issue open; commit 2baafd1)
Task 4: fix round 2/5 (1 addressed, 0 open; commit 647c8b6)
Task 4: complete (selected PTQ matrix, strict hard publication, explicit selected devices and identities, legacy CLI preserved; commits 3a547b7..647c8b6)
Task 5: fix round 1/5 (2 addressed, 2 new provenance issues open; commit f69ec08)
Task 5: fix round 2/5 (2 addressed, 0 open; commit e253b78)
Task 5: complete (contract HAWQ trace, explicit cross-interpreter allocation, strict <=6 budgets and provenance; commits 6820dfd..e253b78)
Task 6: fix round 1/5 (4 original findings addressed, 3 hard-joint/provenance issues open; commit 8998d59)
Task 6: fix round 2/5 (3 addressed, 0 open; commit a7e7175)
Task 6: complete (contract QAT, real hard deployment validation, strict artifacts/resume and CompletionFormer joint owners; commits 843429a..a7e7175)
Task 7: fix round 1/5 (4 original findings addressed, 3 cost/domain/counter issues open; commit ca7d119)
Task 7: Ruling: original implementer became unavailable after context transition; a fresh implementer resumed the preserved uncommitted diff.
Task 7: fix round 2/5 (3 addressed, 0 open; commit b37284c)
Task 7: complete (artifact-bound pooled evaluation, authoritative costs, aligned predictions, paired diagnostics; commits 074df0f..b37284c)
Task 8: fix round 1/5 (CUDA current-device activation, official three-model hard smokes, semantic static-input production/validation, 70-job reproducible DAG, documentation and evidence complete; base 7cbabaa)
Task 8: fix round 2/5 RED (HAWQ canonical output/row enforcement: 2 failed, 9 passed; adapter lifecycle: 1 failed; base 99c6bdc)
Task 8: fix round 2/5 GREEN (80 focused tests passed; 24/24 corrected CUDA records passed propagation/shape/finite/native/official audits; manifests afa28589, 1d87f1e, 3a57a3c7; commit subject `fix: validate HAWQ smoke propagation invariants`)
Task 8: Ruling: bind CompletionFormer to idle physical `cuda:3` for formal execution because GPU 2 has an external 34,144 MiB context; preserve explicit configuration and prohibit runtime fallback (36 tests passed; full eight-method cuda:3 smoke b13774c3).
Task 8: formal run attempt 1 stopped at all three static producers (single-channel initial-depth imbalance was incorrectly selectable with zero IQR; CompletionFormer concat cost hook deleted a closure local).
Task 8: formal static-input fix RED 2 failed; GREEN 49 passed; all three exact 256-candidate producers and 128/64 semantic validators passed with archived receipts d86c3eec, 0157ffd2, d1cc138e.
Task 8: formal run attempt 2 was stopped during P3/T3 after runtime evidence showed the implementation evaluated every prefix x tail Cartesian interaction before choosing P3 (DySPN 186, NLSPN 378, CompletionFormer 2805 interactions), contrary to the approved staged protocol.
Task 8: staged P3/T3 fix RED reproduced the stale QAT artifact contract; GREEN passed 111 affected Python 3.11 tests and 20 P3/T3 tests under the configured Python 3.7 environment. Persisted candidate counts are now DySPN 93, NLSPN 159, CompletionFormer 560, with T3 interactions restricted to the selected P3 prefix.
Task 8: staged-search review found one critical trust-anchor gap and two important execution issues: artifact-owned budgets/costs, a repeated unpersisted T3 baseline, and partial evaluator cleanup on constructor failure.
Task 8: review fix RED reproduced all three issues; GREEN passed 153 affected Python 3.11 tests and 34 P3/T3 tests under Python 3.7. QAT/evaluation now require launch-spec cost/budget anchors, T3 measures interactions only against the persisted P3 baseline, and partial adapters close before constructor exceptions propagate.
Task 8: staged-search fix re-review ACCEPT (original Critical and both Important findings closed; no new Critical/Important; commit 825e833).
