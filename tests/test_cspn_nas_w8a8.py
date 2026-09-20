import json
from pathlib import Path
import sys

import torch

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from cspn_encoder_nas import build_cspn_nas
from scripts import export_nyu_predictions as exporter
from scripts.nyu_quantization_analysis import classify_module
from scripts.rtn_quantization import RTNInstrumentor
from spn_quant.adapters import install_model_semantic_adapter
from spn_quant.nas.spec import EncoderSpec


SPEC = EncoderSpec(32, (32, 64, 128, 256), (1, 1, 1, 0))


def _write_run(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    spec_path = run / "encoder_spec.json"
    spec_path.write_text(json.dumps(SPEC.to_dict()))
    args = {
        "model": "cspn",
        "iteration": 1,
        "cspn_backbone": "resnet18",
        "from_scratch": True,
        "cspn_encoder_spec": "encoder_spec.json",
        "cspn_control_checkpoint": "",
    }
    (run / "args.json").write_text(json.dumps(args))
    (run / "meta.json").write_text(json.dumps({
        "architecture": "CSPN encoder NAS",
        "encoder_spec": SPEC.to_dict(),
    }))
    checkpoint = run / "best.pt"
    torch.save({"net": build_cspn_nas(SPEC, cspn_step=1).state_dict()}, checkpoint)
    return run, checkpoint


def test_exporter_reconstructs_candidate_from_relative_saved_spec(tmp_path):
    run, checkpoint = _write_run(tmp_path)
    args = exporter.load_run_args(run)

    model, metadata = exporter.build_model(args, checkpoint, torch.device("cpu"))

    assert model.encoder_spec == SPEC
    assert Path(args.cspn_encoder_spec) == run / "encoder_spec.json"
    provenance = metadata["model_provenance"]
    assert provenance["encoder_spec"]["slug"] == SPEC.slug
    assert provenance["encoder_spec_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["net"]
    assert torch.equal(model.state_dict()["layer4.0.projection.weight"],
                       saved["layer4.0.projection.weight"])


def test_semantic_adapter_observes_nas_encoder_decoder_heads_and_signals():
    model = build_cspn_nas(SPEC, cspn_step=1).eval()
    adapter = install_model_semantic_adapter(model, "cspn", strict=True)
    adapter.observe()
    with torch.inference_mode():
        model(torch.randn(1, 4, 228, 304))
    adapter.freeze(8)
    rows = {row["site"]: row for row in adapter.semantic_manifest()}
    adapter.close()

    expected = (
        "activation::layer4.0.projection",
        "activation::stem_skip_adapter",
        "activation::stage1_skip_adapter",
        "activation::stage2_skip_adapter",
        "activation::bottleneck_adapter",
        "activation::gud_up_proj_layer1.conv1",
        "activation::gud_up_proj_layer5.conv1",
        "activation::gud_up_proj_layer6.conv1",
        "signal::affinity",
        "signal::propagation_state",
        "signal::prediction",
    )
    assert all(name in rows and rows[name]["observed"] for name in expected)


def test_w8a8_full_groups_candidate_adapters_and_projection_as_encoder():
    model = build_cspn_nas(SPEC, cspn_step=1)
    instrumentor = RTNInstrumentor(
        model, lambda name, module: classify_module("cspn", name, module))
    groups = instrumentor.module_groups()

    for name in (
        "layer4.0.projection", "stem_skip_adapter", "stage1_skip_adapter",
        "stage2_skip_adapter", "bottleneck_adapter",
    ):
        assert groups[name] == "encoder"
