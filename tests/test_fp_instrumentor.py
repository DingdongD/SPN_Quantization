import torch
import torch.nn as nn

from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor


def test_hardware_instrumentor_executes_floating_point_formats():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Conv2d(4, 2, 1)).eval()
    instrumentor = HardwareAlignedInstrumentor(
        model, lambda name, module: "block")
    sample = torch.randn(2, 3, 4, 4)
    instrumentor.observe()
    with torch.no_grad():
        model(sample)
    instrumentor.freeze()
    weights = dict((name, "fp4_e2m1") for name in instrumentor.modules)
    activations = dict(
        (key, "fp8_e4m3fn")
        for key in instrumentor.activation_site_keys({"block"})
        if isinstance(key, tuple))
    instrumentor.configure_floating_point(weights, activations, {"block"})
    with torch.no_grad():
        output = model(sample)
    assert output.shape == (2, 2, 4, 4)
    assert torch.isfinite(output).all()
    assert all(row["format"] == "fp8_e4m3fn"
               for row in instrumentor.manifest()
               if row["kind"] != "weight")
    instrumentor.close()
