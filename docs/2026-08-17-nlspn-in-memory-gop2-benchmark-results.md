# NLSPN Pure-In-Memory GOP2 End-to-End Benchmark Results

## Outcome

The pure-memory causal GOP2 pipeline passed the approved pooled quality gate
but did not accelerate the frozen NLSPN baseline.

- Full NLSPN pooled RMSE: 0.378860 m.
- GOP2 pooled RMSE: 0.381486 m.
- GOP2/full RMSE ratio: 1.006933, or 0.6933% degradation.
- Formal quality gate: pass because 1.006933 is no greater than 1.01.
- Measured end-to-end speedup: 0.526705x.
- Equivalently, GOP2 took 1.8986 times as long as full NLSPN.

The earlier 1.57x estimate was a component projection from staged cached
experiments. It is not supported by this end-to-end measurement.

## Configuration

The benchmark used the eight approved 32-frame clips from
BeachApartmentInterior_My_ir: 0001--0032, 0282--0313, 0563--0594,
0844--0875, 1126--1157, 1407--1438, 1688--1719, and 1969--2000.

- Frames: 256.
- Input geometry: 304 x 228.
- Sparse depth: 500 fixed points per clip, seed 2026.
- Schedule: fixed I, P, I, P with state reset at every clip boundary.
- Warm-up: one complete pass per path.
- Timed repeats: five, totaling 1,280 timed frames per path.
- Precision: FP32.
- GPU: NVIDIA A100-SXM4-40GB.
- Benchmark runtime: PyTorch 1.10.1, NumPy 1.21.6.
- NLSPN: frozen ResNet-34, 18 TGASS propagation steps,
  preserve_input=false.
- NLSPN checkpoint SHA-256:
  bb149132a665b4780d72a5af9d33e4f058dad823090424e55ad0cefa9287a2cb.
- RAFT-Small C_T_V2 weight SHA-256:
  01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27.
- Dynamic fallback: disabled.
- Intermediate tensor cache: disabled.

The end-to-end timing starts with current RGB and sparse depth in CPU memory
and ends after the predicted depth is back in CPU memory. It includes H2D,
model execution, optical flow, warping, propagation, state update, D2H, and
CUDA synchronization. Dataset decoding, model loading, warm-up, report
generation, and disk I/O are outside the steady-state timing.

## Official RAFT-Small Parity

The compatible RAFT-Small has the same 990,162 parameters, strict official
state-dictionary keys, input transforms, 12 updates, and weight artifact as
Torchvision 0.22.1. Within the same PyTorch 2.7 environment, its GPU output is
bitwise identical to the official implementation.

PyTorch 2.7.1 uses cuDNN 9.1, while the NLSPN-compatible PyTorch 1.10.1
environment uses cuDNN 8.2. Their GPU convolution kernels introduce an
approximately 0.006-pixel cross-runtime difference even when deterministic
cuDNN mode is selected. Therefore the cross-runtime architecture/weight
parity gate was evaluated on CPU, while the formal benchmark remained on the
A100 GPU.

- Maximum absolute flow difference: 0.00003053 pixels.
- Flow RMSE difference: 0.00000519 pixels.
- Allowed maximum absolute difference: 0.001 pixels.
- Allowed flow RMSE difference: 0.0001 pixels.
- Parity result: pass.

The parity threshold was not relaxed.

## End-to-End Performance

| Path | Timed frames | Mean (ms) | P50 (ms) | P95 (ms) | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full NLSPN | 1,280 | 13.221 | 12.369 | 17.713 | 75.64 |
| GOP2 overall | 1,280 | 25.100 | 27.819 | 47.103 | 39.84 |
| GOP2 I frames | 640 | 13.538 | 12.812 | 17.052 | 73.86 |
| GOP2 P frames | 640 | 36.663 | 34.000 | 49.229 | 27.28 |

The I-frame time is close to the independent full-NLSPN time, as expected.
The P frame is substantially slower because official RAFT-Small in the legacy
PyTorch/cuDNN environment costs more than the full NLSPN inference it replaces.
Avoiding NPZ cache traffic does not compensate for that compute cost.

Startup and warm-up were reported separately:

- CPU input loading: 10.861 s.
- Model construction/checkpoint loading: 9.083 s.
- Full-path warm-up: 6.962 s.
- GOP2 warm-up: 7.239 s.

Peak CUDA memory:

| Path | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| Full NLSPN | 324.9 MB | 413.1 MB |
| GOP2 | 329.9 MB | 413.1 MB |

GOP2 does not reduce peak reserved memory and increases peak allocated memory
by about 4.9 MB because both frozen models and temporal state remain resident.

## Quality Distribution

All primary RMSE values were independently recomputed from the final CSV over
17,129,486 valid pixels.

| Clip | Full RMSE (m) | GOP2 RMSE (m) | Ratio | 1% gate |
| --- | ---: | ---: | ---: | --- |
| 0001--0032 | 0.661188 | 0.658418 | 0.995810 | pass |
| 0282--0313 | 0.179645 | 0.182859 | 1.017889 | fail |
| 0563--0594 | 0.451889 | 0.456541 | 1.010293 | fail |
| 0844--0875 | 0.332096 | 0.344901 | 1.038557 | fail |
| 1126--1157 | 0.470325 | 0.475566 | 1.011144 | fail |
| 1407--1438 | 0.153354 | 0.153529 | 1.001146 | pass |
| 1688--1719 | 0.172618 | 0.173006 | 1.002249 | pass |
| 1969--2000 | 0.354867 | 0.355803 | 1.002637 | pass |

Of all 256 frames, 62 exceeded a per-frame 1% ratio. The median ratio was
1.0 because all I frames use full NLSPN. P95 was 1.116924 and the worst frame
was 0869 at 1.565200. The pooled pass is therefore not a uniform per-frame or
per-clip guarantee.

## Decision

The experiment confirms that the original frozen NLSPN propagation operator
can causally decode useful temporal residuals while satisfying the approved
pooled 1% RMSE constraint. It also shows that this specific single-process
implementation is not an acceleration: its official RAFT-Small motion
estimator is more expensive than the full NLSPN computation being skipped.

Further speed work should not optimize cache handling. It should target motion
estimation cost, for example a substantially lighter flow estimator or a
non-neural motion approximation, and must re-run the same pooled quality gate.
The present implementation is useful as an accuracy-preserving causal
reference, not as a faster deployment path.

