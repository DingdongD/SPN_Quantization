# NLSPN Temporal Residual Pilot Results

## Configuration

The pilot evaluated 256 frames and 248 causal adjacent-frame pairs from
`BeachApartmentInterior_My_ir`. It used the eight approved 32-frame clips:
0001--0032, 0282--0313, 0563--0594, 0844--0875, 1126--1157,
1407--1438, 1688--1719, and 1969--2000.

- Source geometry: 640 x 480.
- NLSPN geometry: 304 x 228.
- Sparse input: 500 fixed valid locations per clip, seed 2026.
- NLSPN: frozen ResNet-34, 18 propagation iterations, `TGASS`,
  `preserve_input=False`.
- NLSPN checkpoint SHA-256:
  `bb149132a665b4780d72a5af9d33e4f058dad823090424e55ad0cefa9287a2cb`.
- RAFT-Small official weight SHA-256:
  `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`.
- Orchestrator: PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118.
- NLSPN worker: PyTorch 1.10.1 in `completionformer-py37`.
- GPU: NVIDIA A100-SXM4-40GB.

CUDA timing was synchronized around each measured region. Across all clips,
the amortized full NLSPN time was 55.535 ms/frame. RAFT-Small took
11.597 ms/pair and the warmed causal propagation call took 3.700 ms/pair.
The latter excludes file I/O, plotting, cache serialization, and the small
guidance/confidence warp outside the timed propagation region. These numbers
are component measurements, not an end-to-end deployment speed claim.

## Residual Mapping

Flow compensation reduced the pooled adjacent-output residual RMSE from
0.284150 m to 0.209345 m, a factor of 0.73674 (26.33% lower). It reduced
residual RMSE on 87.90% of the 248 individual pairs. The in-bounds flow
coverage averaged 98.67%.

After alignment:

- 58.01% of valid pixels had an absolute output residual at or below 1 cm;
- 74.72% were at or below 2 cm;
- 86.14% were at or below 5 cm;
- 91.08% were at or below 10 cm;
- median absolute residual was 0.00740 m;
- P95 absolute residual was 0.22282 m.

The current sparse residual had a strong relationship with the aligned full
NLSPN output residual: mean pairwise Pearson correlation was 0.8416 and mean
Spearman correlation was 0.7157. The RGB photometric residual was much weaker
(Pearson 0.1674, Spearman 0.1941), while flow magnitude alone was effectively
uncorrelated (-0.0256 and -0.0226). The observed mapping is therefore driven
primarily by current sparse-depth anchors plus learned NLSPN affinities, not
by motion magnitude alone.

## Existing Propagation Reconstruction

All values below are pooled over 16,588,448 valid target pixels and were
independently recomputed from the saved per-clip arrays.

| Path | GT RMSE (m) | Ratio to full NLSPN | Gate |
| --- | ---: | ---: | --- |
| Full per-frame NLSPN | 0.378609 | 1.000000 | reference |
| Current-guidance propagation oracle | 0.373420 | 0.986294 | pass |
| Warped-history causal propagation | 0.381014 | 1.006351 | pass |

The causal path increases pooled RMSE by 0.6351%, which is below the approved
1% limit. The current-guidance oracle improves RMSE by 1.3706%, showing that
the original frozen propagation operator can express a useful signed
residual correction. No residual network, fine-tuning, or dynamic fallback
was used.

For the intended fixed `I, P, I, P` schedule, using full NLSPN on even local
frames and the causal reconstruction on odd local frames gives a pooled RMSE
ratio of 1.007170 over the evaluated target frames. From the measured compute
components, this schedule projects to 35.416 ms/frame versus 55.535 ms/frame,
or about 1.57x. This is a projection rather than an end-to-end benchmark.

The pooled pass hides meaningful local variation:

| Clip | Oracle ratio | Causal ratio |
| --- | ---: | ---: |
| 0001--0032 | 0.991983 | 0.995686 |
| 0282--0313 | 1.016724 | 1.027698 |
| 0563--0594 | 0.954627 | 0.989744 |
| 0844--0875 | 0.998265 | 1.060690 |
| 1126--1157 | 0.996996 | 1.017621 |
| 1407--1438 | 1.001255 | 1.000582 |
| 1688--1719 | 1.004873 | 1.007970 |
| 1969--2000 | 0.978248 | 0.998628 |

Of the 248 individual causal frames, 117 exceeded a per-frame 1% ratio. The
worst was frame 0869 at 1.56518. The pilot's agreed gate was pooled RMSE, so
these local failures do not change the formal pass, but they rule out a claim
that quality is uniformly preserved frame by frame.

## Decision

The `warped-history causal` branch passes the approved pooled RMSE gate. The
experiment supports the hypothesis that adjacent frozen-NLSPN outputs have a
motion-compensated residual mapping and that the existing NLSPN propagation
operator can decode a useful residual from current sparse-depth differences.

A separate fixed-GOP codec implementation design is therefore justified.
That design should keep the tested `GOP=2` schedule and should treat the
per-clip and worst-frame variability as an explicit limitation. Because the
user excluded automatic fallback, the next stage must report both pooled and
local quality distributions rather than representing the 0.635% pooled
increase as a uniform guarantee.
