# NLSPN Frame-Difference Cache Pilot Results

## Outcome

The three causal, no-RAFT variants accelerated 256-frame GOP2 inference, but
none met the approved all-frame pooled RMSE ratio limit of 1.01. The result is
therefore negative under the requested "preserve accuracy within 1%" gate.

| Path | RMSE (m) | RMSE ratio | <=1% | Mean ms/frame | P ms/frame | Speedup |
|---|---:|---:|:---:|---:|---:|---:|
| Full NLSPN | 0.378860 | 1.000000 | pass | 10.310 | n/a | 1.000x |
| Zero-flow residual | 0.394604 | 1.041556 | fail | 7.063 | 3.517 | 1.460x |
| RGB-difference cache | 0.394604 | 1.041558 | fail | 7.510 | 4.427 | 1.373x |
| Global translation + difference cache | 0.387728 | 1.023407 | fail | 8.450 | 6.175 | 1.220x |

Global translation recovered roughly half of the zero-flow quality loss, but
remained 2.34% above the full-reference RMSE. RGB-difference output caching
was effectively quality-neutral relative to zero-flow and made inference
slower.

## Frozen protocol

- Dataset: `BeachApartmentInterior_My_ir`, eight independent 32-frame clips,
  256 frames total, preprocessed to 304 x 228.
- Input: identical RGB and 500 deterministic sparse-depth points per frame,
  seed 2026.
- Model: frozen ResNet-34 NLSPN, TGASS, 18 propagation iterations,
  `preserve_input=false`.
- Checkpoint SHA-256:
  `bb149132a665b4780d72a5af9d33e4f058dad823090424e55ad0cefa9287a2cb`.
- Calibration: clips 0001-0032, 0282-0313, 0563-0594, and 0844-0875.
- Held-out: clips 1126-1157, 1407-1438, 1688-1719, and 1969-2000.
- Timing: one warm-up followed by five timed repeats for each final path;
  CPU-memory input through CPU-memory prediction.
- Hardware: NVIDIA A100-SXM4-40GB; measured peak allocated CUDA memory was
  316,582,400 bytes and peak reserved memory was 400,556,032 bytes.
- No RAFT construction, fine-tuning, automatic fallback, or intermediate
  tensor files were used.

The RGB and global sweeps both selected threshold `2/255` and dilation radius
8 using calibration data only. Their calibration RMSE values were 0.459311 m
and 0.448580 m respectively.

## Calibration and held-out quality

| Path | Calibration ratio | Held-out ratio | All-frame ratio |
|---|---:|---:|---:|
| Zero-flow residual | 1.048753 | 1.026116 | 1.041556 |
| RGB-difference cache | 1.048755 | 1.026118 | 1.041558 |
| Global translation + difference cache | 1.024252 | 1.021612 | 1.023407 |

The held-out result confirms that the failure is not caused only by selecting
parameters on the calibration clips. All three variants exceed 1.01 on both
partitions.

## Cache and latency analysis

On repeat-zero P frames, the selected RGB mask marked only 1.799% of pixels
stable and the selected global-translation mask marked 1.688% stable. The
corresponding changed fractions were 98.201% and 98.312%. Sparse-depth
inconsistency masks covered about 31.1% and 31.3% after dilation. Global
translation added a 0.452% mean out-of-bounds region.

These masks do not reduce the current implementation's dense propagation
work: the unchanged/current blend is applied after the original NLSPN
`prop_layer` has produced a full-frame residual candidate. Consequently:

- zero-flow is fastest because it skips motion and mask work;
- RGB difference adds about 0.91 ms to each P frame without improving quality;
- phase correlation and four state warps add about 2.66 ms per P frame over
  zero-flow, while improving quality but not enough to pass the gate.

The observed acceleration comes from omitting the NLSPN backbone on P frames,
not from selectively avoiding propagation in unchanged regions. Achieving
additional cache-based compute savings while retaining the exact model would
require a sparse/tiled execution mechanism inside or around the propagation
operator; output blending alone cannot provide that saving.

For comparison, the earlier RAFT-GOP2 run achieved RMSE ratio 1.006933 but was
slower than full NLSPN at 0.526705x. The current global-translation method is
faster at 1.220x, but fails the quality gate.

## Integrity checks

Independent recomputation from `frame_metrics.csv` reproduced every split
ratio and speedup in `summary.json`. The final directory contains exactly six
nonempty report artifacts, 24 calibration rows, 5,120 timed frame rows, and no
NPY/NPZ prediction, flow, guidance, confidence, FFT, or cache files.

Formal artifacts:
`/workspace/VoxelNet/nlspn_frame_difference_cache/BeachApartmentInterior_My_ir/pilot_256`
