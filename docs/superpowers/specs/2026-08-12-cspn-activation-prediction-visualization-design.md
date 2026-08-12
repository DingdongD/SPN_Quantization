# CSPN Activation Prediction Visualization Design

## Scope

Render the existing audited CSPN activation-resolution predictions without
running inference again. The input contract is exactly four configurations,
each containing the same 64 sample indices:

- `FP32`;
- `W4A4_RTN`;
- `W4A4_CHANNEL`;
- `W4A4_CALIBRATED_SCALE`.

## Outputs

The script writes two PNG figures and matching PDF files under the experiment
`figures/` directory:

- a four-sample detail comparison;
- a 64-sample contact sheet.

Each detail row contains RGB, sparse depth, GT, FP32, RTN W4A4, per-channel
W4A4, calibrated-scale W4A4, and one absolute-error map for each prediction.
The contact sheet contains GT and the four predictions for every sample.

## Visual Contract

All depth panels use the same 0-10 m scale. Error panels share the global P99
absolute-error scale computed over valid pixels from all 64 samples and three
quantized configurations. Invalid GT pixels are masked. Arial is requested
with Liberation Sans and DejaVu Sans as Matplotlib font substitutes.

The four detail samples are selected deterministically as RTN worst, largest
per-channel improvement over RTN, RTN median, and per-channel worst. Duplicate
indices are skipped and replaced by the next ranked sample.

## Validation

The loader rejects missing or extra configuration directories, mismatched
sample indices, inconsistent GT/RGB/sparse payloads, incomplete keys, and
non-finite prediction values. Tests cover the strict contract, deterministic
selection, and figure creation. Generated figures remain outside Git.
