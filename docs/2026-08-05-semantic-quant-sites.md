# Semantic Quantization Sites

This change introduces `QuantSpec` and `QuantSite` as the stable contract
between model adapters, calibration, QDQ simulation, and manifests.

A site represents one logical tensor edge. A producer with multiple consumers
is registered once with an explicit fan-out list, rather than being quantized
again at every consumer hook. The default low-bit activation contract is
uniform signed W4A4 (`transform=none`). LogNP remains available only as an
explicit per-site transform for controlled ablations.

The execution tracer uses call-indexed names (`module#0`, `module#1`, ...), so
reused recurrent modules do not silently share calibration state. Registries
fail closed on duplicate and unknown site names and can be frozen before an
experiment begins.
