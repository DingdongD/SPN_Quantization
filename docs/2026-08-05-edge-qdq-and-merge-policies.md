# Logical-Edge QDQ and Merge Policies

`EdgeQDQRuntime` prevents producer-output, consumer-input, and fan-out hooks from
quantizing the same tensor repeatedly in one forward. A new quantization is
allowed only at an explicit requantization boundary.

Add and Concat adapters now support three policies:

- `shared`: one scale across branches; retained as the standard-backend baseline.
- `independent`: each branch keeps its own scale before wide-domain Add/Concat.
- `grouped`: channel-group scales are used for the merged Concat output or are
  shared group-wise across Add branches.

Merge manifests record policy, operation, branch count, scale layout, group
size, and bit width. Adapters can assert the expected number of executed merge
sites and fail closed when model adaptation is incomplete.

The edge-aware compatibility wrapper currently applies to uniform quantizers.
LogNP is intentionally excluded from automatic edge reuse and remains an
explicit ablation transform rather than a default W4A4 policy.
