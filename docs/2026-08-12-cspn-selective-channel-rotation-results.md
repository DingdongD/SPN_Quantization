# CSPN Selective Channel Rotation Results

## Protocol

- Model: official CSPN ResNet-18 architecture from the converged
  `cspn_iter24/best.pt` checkpoint.
- Data: NYU Depth V2, 128 fixed training samples for calibration and the
  existing fixed 64-sample evaluation set.
- CNN: signed per-output-channel W4 weights and A4 activations.
- Rotation boundaries: signed decoder entry and layer4 signed skip only.
- Exclusions: ReLU outputs and the complete guidance head remain unrotated;
  the guidance head remains FP32.
- Propagation: affinity A8, state A8, and signed INT16 Q13 coefficients.
- Bias: FP32 because a group-wise boundary activation does not define one
  input scale for integer bias rescaling.
- Weight order: rotate the folded FP weight first, then quantize the
  transformed weight to the signed W4 grid.

Group size selection used only calibration block-output error. Group size 16
was selected from 16, 32, and 64.

## End-to-End Results

| Configuration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Flat RMSE (m) | Boundary RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.166932 | 0.067154 | 0.022361 | 0.022990 | 0.115248 | 0.416973 |
| RTN W4A4 | 0.439264 | 0.331009 | 0.150977 | 0.099151 | 0.419504 | 0.563371 |
| Group W4A4 | 0.426802 | 0.318601 | 0.146784 | 0.096751 | 0.406064 | 0.560208 |
| Random, decoder entry | 0.439497 | 0.331280 | 0.151180 | 0.099467 | 0.419574 | 0.567080 |
| Random, layer4 skip | 0.436762 | 0.328743 | 0.150149 | 0.094923 | 0.416597 | 0.565406 |
| Random, both | 0.436364 | 0.329076 | 0.150382 | 0.095110 | 0.416245 | 0.567205 |
| Hadamard, decoder entry | 0.440080 | 0.332095 | 0.151663 | 0.099395 | 0.420429 | 0.563823 |
| Hadamard, layer4 skip | 0.435859 | 0.328308 | 0.149269 | 0.094270 | 0.415758 | 0.565915 |
| Hadamard, both | 0.436468 | 0.328958 | 0.149690 | 0.094381 | 0.416490 | 0.565287 |
| Hadamard + Group, both | 0.430580 | 0.324733 | 0.147494 | 0.093369 | 0.409937 | 0.563930 |

Hadamard + Group improves RMSE by 0.008685 m (1.98%) over RTN W4A4, but is
0.003778 m (0.89%) worse than Group W4A4. Hadamard on the layer4 signed skip
alone improves RMSE by 0.003405 m (0.78%) over RTN.

## Boundary Analysis

At the layer4 signed skip, Hadamard lowers P99.99 magnitude from 1.76198 to
1.58714 and raises A4 SQNR from 1.62 dB to 4.12 dB. Adding Group-A4 raises it
to 4.69 dB. This confirms that rotation redistributes the problematic signed
skip channels.

At decoder entry, Hadamard raises P99.99 magnitude from 2.74340 to 3.11426 and
reduces A4 SQNR from 11.44 dB to 9.49 dB. Group-A4 partly recovers the SQNR to
12.37 dB, but the block-output SQNR remains effectively unchanged. Decoder-entry
rotation is therefore not a useful CSPN boundary under this calibration.

The selective rotation hypothesis is only partially supported. Rotation helps
the layer4 signed skip locally, but does not outperform the simpler Group-A4
baseline end to end. Learned Rotation or Rotation + BRECQ + QDrop should not be
continued from this boundary configuration without first revising the selected
rotation sites.

## Artifacts

Runtime outputs are stored outside Git under:

```text
/workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/
```

The directory contains the complete metric tables, 640 prediction payloads,
and the RMSE, activation, and prediction comparison figures. Reproduce the
evaluation with:

```bash
PYTHONPATH=. python scripts/run_nyu_cspn_rotation.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint best.pt \
  --sample-metrics /workspace/SPN_Quantization/profile_logs/nyu_propagation_aware_quantization_unified/cspn/sample_metrics.csv \
  --data-root /workspace/CSPN/cspn_pytorch \
  --device cuda:0 \
  --calibration-samples 128 \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4

PYTHONPATH=. python scripts/plot_nyu_cspn_rotation.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/cspn \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/analysis
```
