# CSPN Stratified Calibration Results

## Run

The train-only selector ran against the official 6,700-sample NYU CSPN train
split and converged `cspn_iter24/best.pt` checkpoint. It reserved 512 audit
samples, selected 1,024 candidates from the remaining 6,188 eligible samples,
and produced 32 tail plus 96 density-weighted k-medoids calibration samples.
No validation or test samples and no depth prediction metrics participated in
selection.

The run passed the declared coverage contract. Its immutable checkpoint SHA256
is `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.

## Coverage

| Metric | Stratified 128 | Current random 128 | 16-random mean |
| --- | ---: | ---: | ---: |
| Audit nearest distance p50 | 0.14803 | 0.13791 | 0.14045 |
| Audit nearest distance p95 | 0.27597 | 0.27288 | 0.30104 |
| Audit nearest distance maximum | 16.38321 | 22.59497 | 22.09791 |
| Mean scalar range coverage | 99.71% | 98.18% | 98.60% |
| Uncovered activation maxima | 2/6 | 6/6 | not an acceptance target |

The stratified p95 is lower than the predeclared random-baseline mean, although
the current random seed is slightly lower. The stratified set improves tail and
worst-case coverage rather than central-distribution matching: its p50 is higher
and its standardized mean scalar Wasserstein distance is 0.33850 versus 0.07467
for the current random set. This is consistent with reserving 25% of the final
budget for explicit p5/p95 tail coverage.

The two remaining audit activation maxima outside the calibration range are
`decoder_layer2_fusion_max` with ratio 1.19915 and
`decoder_layer4_relu_max` with ratio 1.07969. Encoder stem convolution, stem
ReLU, layer-1 ReLU, and decoder layer-4 fusion maxima are covered.

## Integrity

- Calibration indices: 128 unique.
- Audit indices: 512 unique and disjoint from calibration.
- Candidate representative-weight sum: 6,188.
- Final representative-weight sum: 6,188.
- Final composition: 9 tail-cover, 23 tail-score, and 96 medoid samples.
- Output contract: exactly 11 tabular/JSON/Markdown artifacts and no images.

Artifacts are stored at
`/workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128`.
