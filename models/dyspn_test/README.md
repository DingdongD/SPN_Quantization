# DySPN HW-aligned Models

This directory contains the initial DySPN compiler-aligned model scaffold.

## Strict Modules

- `dyspn_hw_aligned.py`: HW-aligned DySPN architecture.
- `dyspn_hw_guide.py`: RGBD -> guide subgraph.
- `dyspn_hw_offset_aff.py`: guide -> offset/aff Conv subgraph. Use this for real deployment with `DYSPN_HW_CKPT`.

## Probe-only Module

- `dyspn_hw_offset_aff_compile_probe.py`: no-checkpoint compile smoke test.

The strict DySPN initialization leaves `conv_offset_aff` at all zeros. The legacy quantizer cannot handle all-zero activation range, so the probe injects nonzero random weights only to validate the Conv shape. Do not use it as a semantic deployment model.

