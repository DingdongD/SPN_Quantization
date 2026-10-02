# Hardware-aware NAS strategy for SPN depth completion

## Optimization target

For architecture `a` and precision assignment `q`, the primary feasibility
constraint is

```text
RMSE(a, q) / RMSE(vanilla FP32) <= 1.02.
```

Among feasible candidates, optimization is lexicographic:

1. measured mapped-resident U250 end-to-end latency;
2. packed parameter bytes, including actual per-layer bit widths;
3. parameter count and MAC count;
4. region count and boundary-layout conversion cost.

Software GPU latency is diagnostic only. It is not used as a substitute for
U250 latency because framework launch overhead, deformable propagation, and
physical layout conversions can reverse the ordering predicted by parameter
count.

## Search-space decomposition

The search is factorized into three ordered spaces:

1. **Depth topology:** remove repeated encoder or Transformer blocks while
   keeping the pretrained stage interfaces.
2. **Interface-preserving width:** reduce decoder bridges, MLP hidden widths,
   and residual-branch hidden channels without changing skip, concatenation,
   or propagation-head tensor shapes.
3. **Precision:** calibrate W8A8 first, then selectively lower insensitive
   weights or activations to W6/W4/A6/A4 while retaining sensitive boundaries
   in BF16 and propagation state in BF16 with FP32 accumulation.

This ordering limits interaction complexity. A precision search is performed
on the final structural candidate rather than assuming that a precision map
selected for the vanilla model remains optimal after NAS.

## Channel inheritance

For a removable hidden channel `c` between an incoming operator, normalization,
and outgoing operator, channels are ranked without validation-set fitting:

```text
score(c) = ||W_in[c]||_2 * ||W_out[:, c]||_2 * |gamma[c]|.
```

The selected channel indices are sorted before slicing so their original order
is preserved. Both sides of the hidden dimension and the corresponding batch
normalization state are sliced together. Candidate widths use hardware-friendly
multiples of 64: 448, 384, and 320 for a 512-channel source.

The method deliberately preserves public tensor widths. It avoids inserting
gather/scatter adapters at residual additions and decoder concatenations, which
would reduce the likelihood that parameter savings become U250 latency gains.

## Multi-fidelity protocol

1. Evaluate all candidates on the same deterministic 64-image screening set.
2. Reject candidates with unstable output direction, using paired output cosine
   in addition to RMSE.
3. Promote only Pareto candidates to the full 654-image validation set.
4. Re-run calibration-derived W8A8 and selected low-bit assignments on the
   promoted structure.
5. Accept only the combined NAS, pruning, and quantization result that satisfies
   the 2% vanilla-relative RMSE constraint.
6. Lower accepted candidates and measure them with the same mapped-resident
   runner, sample cohort, synchronization, and physical-layout policy.

Calibration uses a frozen 128-image cohort. Scales are derived from calibration
statistics; validation-set fitting and manually chosen scale overrides are not
part of the search.

## Second-round screening evidence

The table reports reduction relative to the already selected depth-NAS model.

| Model | Added search dimensions | Selected candidate | RMSE change | Output cosine | Added parameter reduction |
| --- | --- | --- | ---: | ---: | ---: |
| DySPN | bridge width + stage5 residual hidden width | bridge 62.5%, hidden 62.5% | -0.402% | 0.999993 | 20.16% |
| NLSPN | bridge width + stage5 residual hidden width | bridge 62.5%, hidden 62.5% | -0.322% | 0.999986 | 20.57% |
| CompletionFormer | all MLP hidden widths + PVT stage4 CNN hidden width | MLP 62.5%, hidden 62.5% | -0.094% | 0.999996 | 13.43% |

These improvements are screening observations, not evidence that pruning is a
general accuracy regularizer. The defensible conclusion is narrower: the
removed low-importance hidden channels are redundant on the screening cohort,
so the candidates merit full-validation promotion.

## Transferable findings

1. **Depth and hidden width are complementary.** Removing repeated blocks and
   narrowing the surviving blocks target different parameter populations.
2. **Deep hidden dimensions are safer than task interfaces.** Internal residual
   and MLP widths can be reduced while preserving skip and decoder contracts.
3. **Sensitivity follows task boundaries, not parameter count.** Initial depth,
   guidance, encoded features, and propagation inputs deserve higher precision
   even when their parameter contribution is small.
4. **Weight and activation precision must be separated.** CSPN shows that large
   W4 weight coverage can add almost no error while encoded A8 boundaries remain
   the dominant quantization source.
5. **Hardware alignment belongs in the search space.** Channel counts should
   match compiler tiling and vector widths, and region/layout costs must be
   measured rather than repaired after selecting an unconstrained architecture.
6. **Compression is not acceleration.** A candidate is called faster only after
   identical mapped-resident board measurements; parameter or MAC reduction is
   reported separately.

## Ablations required for an article

- depth-only versus width-only versus depth-plus-width;
- magnitude-only versus joint incoming/outgoing/normalization importance;
- arbitrary width versus U250-aligned width;
- W8A8 versus selected mixed precision on the same final structure;
- inherited weights versus short recovery fine-tuning;
- software latency versus CModel and mapped-resident board latency;
- parameter/MAC reduction versus region count and layout-conversion overhead.

Every table should identify the vanilla checkpoint, dataset indices, metric
aggregation, calibration cohort, propagation precision, compiler build, board
runner, and whether latency includes host transfer and synchronization.

## Full-validation Pareto results

The promoted candidates were evaluated on all 654 NYU validation images with
the frozen 128-image calibration cohort. The robust deployment points are:

| Model | Final low-bit RMSE | Relative to vanilla | Parameter reduction | Packed-weight compression |
| --- | ---: | ---: | ---: | ---: |
| CSPN | 0.145823 m | +1.307% | 61.64% | 15.21x |
| DySPN | 0.107021 m | +0.399% | 40.95% | 8.21x |
| NLSPN | 0.118157 m | +1.638% | 41.33% | 7.28x |
| CompletionFormer | 0.109132 m | +0.850% | 44.76% | 9.31x |

DySPN also has an aggressive 47.28%-parameter-reduction point at +1.863%, but
its 0.137-percentage-point gate margin is too small for the default deployment
choice. NLSPN's corresponding aggressive candidate failed screening, showing
that family-specific sensitivity matters even when modules share a ResNet-like
shape.

The small-set and full-set ordering are not identical. In particular, the
aggressive DySPN candidate looked neutral on 64 images but moved to +1.419% in
FP32 on the full set. Therefore the multi-fidelity protocol uses screening only
for promotion and reserves every accuracy claim for the complete cohort.

These are software accuracy and model-storage results. Hardware speedup remains
an open measurement until lowering preserves the selected dimensions and the
models run through the same mapped-resident U250 runner.

The table uses the unified BF16-state/FP32-accumulator propagation contract.
Relative to the preceding CSPN-FP32/DySPN-NLSPN-CompletionFormer-FP16 state
choices, BF16 state adds 0.141%, 0.189%, 0.433%, and 0.520% RMSE respectively.
This supports a common BF16 propagation-state interface, but it is not evidence
for pure BF16 accumulation. NLSPN's two legacy floating-point protected
boundaries also require explicit BF16 QDQ validation before the complete model
can be described as an INT4/INT6/INT8/BF16-only graph.
