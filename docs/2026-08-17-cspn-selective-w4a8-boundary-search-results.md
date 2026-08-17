# CSPN Selective W4A8 Boundary Search Results

## Protocol

- Official CSPN ResNet-18, 24 propagation iterations.
- Checkpoint: `output/nyu_converged_baselines/cspn_iter24/best.pt`.
- Calibration: persisted stratified 128-sample NYU train subset.
- Evaluation: persisted fixed 64-sample NYU validation subset, seed
  `20260812`.
- Primary weights: Group-8 W4 except W8
  `gud_up_proj_layer5.conv1`.
- Primary activations: Group-8 A4/A8 factorial over stem, encoder layer1,
  encoder layer2, and decoder layer4.
- Guidance: FP32. Propagation: A8 state/affinity/confidence, INT16 Q13
  coefficients, and INT32 accumulation.
- Required aggregate RMSE: at most 0.1773 m.

## Outcome

No primary Stage-1 candidate satisfies the fixed RMSE target. The runner
therefore stopped before Stage 2 as required and did not create a winner or a
feasible Pareto frontier.

The best primary candidate is `ACT_MASK_15`, with all four searched activation
units at A8:

| Configuration | RMSE (m) | MAE (m) | A8 element fraction | W8 MAC fraction | Normalized added bit cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict W4A4 context | 0.342178 | 0.257888 | 0.000000 | 0.000000 | 0.000000 |
| Fixed-head all-A4 (`ACT_MASK_00`) | 0.335302 | 0.251500 | 0.000000 | 0.002665 | 0.000012 |
| Best primary (`ACT_MASK_15`) | 0.202234 | 0.121170 | 0.568420 | 0.002665 | 0.291957 |
| P3/T3 W8A8 context | 0.172321 | 0.086617 | 0.746503 | 0.463250 | 0.404034 |

`ACT_MASK_15` misses the target by 0.024934 m. All its predictions are finite
and positive, and its coefficient-sum error, contraction violation rate, and
anchor error are zero. Its rejection is caused only by RMSE.

## Unit Effects

The factorial mean RMSE changes from switching each unit from A4 to A8 are:

| Unit | Mean RMSE change (m) |
| --- | ---: |
| Decoder layer4 | -0.069472 |
| Stem | -0.055820 |
| Encoder layer1 | -0.015705 |
| Encoder layer2 | -0.002816 |

The strongest recovery is produced by the stem/decoder4 combination. With all
other searched units A4, stem alone reaches 0.312370 m, decoder4 alone reaches
0.278720 m, and stem plus decoder4 reaches 0.208534 m. This non-additive gain
shows that the RGBD entrance precision and final decoder reconstruction
precision interact across the encoder-decoder path.

Adding encoder layer1 and layer2 A8 to the stem/decoder4 candidate only lowers
RMSE from 0.208534 m to 0.202234 m. The remaining error is therefore not
recoverable by these four activation-unit promotions under the all-ordinary-W4
weight contract.

## Interpretation

The P3/T3 context reproduces the prior 0.172321 m result, proving that the
checkpoint, dataset protocol, and quantization instrumentation are consistent.
It differs from `ACT_MASK_15` in two coupled ways: it has 29 A8 activation
owners rather than the primary candidate's searched owner set, and it promotes
15 Conv weights to W8 (46.3250% of Conv MACs) rather than only the initial-depth
head. The 0.029912 m gap cannot be assigned solely to weights without a new
controlled ablation.

The next controlled search should start from `ACT_MASK_15` and independently
test the missing initial-depth activation owner and the P3/T3 W8 weight units.
That search changes the immutable primary-weight contract and is not folded
into this result.

## Artifacts

Failure diagnostics are retained at
`profile_logs/nyu_cspn_selective_w4a8_boundary_search_64.incomplete`:

- `stage1_aggregate_metrics.csv`
- `stage1_sample_metrics_64.csv`
- `stage1_propagation_metrics.csv`
- `activation_cost_basis.csv`
- `stage1_unit_mask_rmse.png`
- `stage1_unit_mask_rmse.pdf`

The `.incomplete` suffix is intentional: Stage 2, prediction reruns, the
feasible Pareto frontier, and final artifact publication were not executed
because no Stage-1 anchor met the target.
