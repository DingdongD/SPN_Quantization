# CSPN Decoder and Initial-Depth Sensitivity Results

## Protocol

This experiment uses the official CSPN ResNet-18 model, the converged
`cspn_iter24/best.pt` checkpoint, 24 propagation iterations, the persisted
128-sample stratified NYU train calibration set, and the fixed 64-sample NYU
validation set. The strict baseline is static contiguous Group-8 W4A4 MinMax.
Guidance remains FP32 and propagation remains A8/INT16-Q13/INT32. Every tested
configuration starts from a fresh checkpoint load and performs a fresh
calibration pass; no training or evaluation-driven calibration is used.

The run evaluated 58 configurations and 3,712 sample/configuration pairs. All
metrics were finite. Nine selected configurations contain all 64 prediction
payloads, and all 590 manifest artifact hashes were verified after plotting.

## Block Search

| Block configuration | RMSE (m) | Delta (m) | MAE (m) | AbsRel | Added cost |
|---|---:|---:|---:|---:|---:|
| Strict W4A4 | 0.342178 | 0.000000 | 0.257888 | 0.113045 | 0.000% |
| Decoder 4 W8A8 | 0.284687 | -0.057490 | 0.205959 | 0.084901 | 18.943% |
| Decoder 4 W4A8 | 0.286173 | -0.056005 | 0.206817 | 0.085283 | 18.293% |
| Decoder 3 W8A8 | 0.321402 | -0.020776 | 0.239410 | 0.100215 | 7.361% |
| Decoder 3 W4A8 | 0.325568 | -0.016609 | 0.242503 | 0.101162 | 6.288% |
| Initial depth W8A4 | 0.335302 | -0.006876 | 0.251500 | 0.111028 | 0.001% |
| Decoder 2 W8A8 | 0.337614 | -0.004564 | 0.254672 | 0.111167 | 7.490% |
| Decoder 1 W4A8 | 0.343071 | +0.000894 | 0.258787 | 0.113362 | 0.771% |
| Decoder 4 W8A4 | 0.356059 | +0.013882 | 0.271384 | 0.122225 | 0.650% |

Decoder 4 and decoder 3 were selected for the site search because their best
block configurations produced the lowest RMSE. The dominant decoder failure is
activation quantization: decoder 4 W4A8 recovers 97.4% of the RMSE reduction
obtained by decoder 4 W8A8, while decoder 4 W8A4 is worse than strict W4A4.
Decoder 3 shows the same ordering, although W8 weights provide an additional
gain once its activations are A8.

The initial-depth head has the opposite behavior. Its W4A8 result is identical
to strict W4A4, while W8A4 lowers RMSE by 0.006876 m. W8A8 is identical to
W8A4. Therefore, the `gud_up_proj_layer5.conv1` weight is sensitive but its
input activation is not. This promotion costs only 0.0012% normalized added
bit-elements and covers 0.2665% of executed Conv MACs.

## Site Search

The strongest individual activation site is
`gud_up_proj_layer3.sc_conv1::output`. Promoting only this decoder-3 shortcut
branch output to A8 lowers RMSE from 0.342178 m to 0.328113 m at 0.572% added
cost and 1.113% A8 activation elements. The next strongest site is
`gud_up_proj_layer4.relu#0::relu_output`, the main branch immediately before
the decoder-4 skip concatenation. It reaches 0.334695 m at 2.287% added cost.

The strongest individual weight site is
`gud_up_proj_layer3.conv1_1`, which reaches 0.338333 m with W8 weights. Its
W8A8 site configuration has the same RMSE, so promoting the registered input
edge does not provide an additional benefit at that site. Decoder-4 individual
weight promotions all regress; decoder-4 accuracy is recovered by coordinating
multiple activation boundaries rather than by isolating one Conv weight.

## Error Path

| Configuration | Decoder-4 MSE | Decoder-4 SQNR | Initial-depth MSE | Propagation MSE |
|---|---:|---:|---:|---:|
| Strict W4A4 | 0.044490 | 5.436 dB | 0.294950 | 0.091423 |
| Initial depth W8A4 | 0.044490 | 5.436 dB | 0.284084 | 0.086721 |
| Decoder 4 W4A8 | 0.019743 | 8.965 dB | 0.163701 | 0.055169 |
| Decoder 4 W8A8 | 0.019470 | 9.025 dB | 0.134918 | 0.054896 |
| Cumulative 08 | 0.036721 | 6.270 dB | 0.228700 | 0.072629 |

The decoder-4 activation recovery reduces local decoder error before the depth
head, then reduces both initial-depth and final propagation error. This is
evidence of upstream activation noise entering the depth head and subsequently
being carried through CSPN propagation. The propagation implementation itself
is unchanged across candidates, so the measured recovery does not come from
changing SPN arithmetic.

RMSE, MAE, AbsRel, flat RMSE, and boundary RMSE improve consistently for the
best decoder-4 configurations. iRMSE is much less stable because isolated
near-zero predictions dominate inverse-depth error; several partial promotions
increase iRMSE despite reducing RMSE. Candidate selection therefore follows
the predefined RMSE objective, but deployment should reject a configuration
that fails an explicit iRMSE or minimum-depth constraint.

## Pareto Result

`CUMULATIVE_08` is the best measured cumulative site configuration: RMSE
0.313868 m, 8.428% added cost, 14.469% A8 activation elements, 2.052% W8 weight
elements, and 13.980% W8 Conv MACs. It promotes the decoder-3 shortcut output,
the decoder-4 first and final ReLU outputs, and three decoder-3 Conv weights
plus selected decoder-3 input boundaries. Further cumulative activation
promotions do not change RMSE and are dominated.

For the lowest-cost operating point, promote only the initial-depth weight to
W8. For a stronger mixed point, use the measured cumulative set through
`CUMULATIVE_08`. For maximum recovery within this search boundary, keep the
entire decoder-4 activation block at A8; adding decoder-4 W8 weights yields only
another 0.001486 m RMSE improvement at 0.650% additional normalized cost.

The complete measured tables, prediction payloads, manifest, and Arial Pareto
plot are stored in
`profile_logs/nyu_cspn_decoder_depth_head_sensitivity_64`.
