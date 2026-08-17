import numpy as np
import torch

from scripts import run_spn_sequence_worker as worker


def test_model_namespace_matches_converged_baseline_arguments():
    args = {
        "iteration": 18,
        "from_scratch": False,
        "lr": 0.001,
        "nlspn_network": "resnet34",
        "completionformer_model": "CompletionFormer",
    }
    nlspn = worker.nlspn_namespace(args)
    assert (nlspn.network, nlspn.prop_time, nlspn.prop_kernel) == (
        "resnet34", 18, 3)
    assert (
        nlspn.conf_prop, nlspn.affinity, nlspn.affinity_gamma) == (
            True, "TGASS", 0.5)
    assert nlspn.preserve_input is False
    completionformer = worker.completionformer_namespace(args)
    assert completionformer.model == "CompletionFormer"
    assert completionformer.max_depth == 10.0


class DySPNToy(torch.nn.Module):
    def forward(self, rgb, dep):
        return dep + rgb[:, :1]


class DictToy(torch.nn.Module):
    def forward(self, sample):
        return {"pred": sample["dep"] + sample["rgb"][:, :1]}


def test_predict_frames_adapts_dyspn_and_dict_model_signatures():
    rgb = np.ones((2, 3, 4, 6), dtype=np.float32)
    dep = np.full((2, 4, 6), 2.0, dtype=np.float32)
    for model_name, model in (
            ("dyspn", DySPNToy()),
            ("nlspn", DictToy()),
            ("completionformer", DictToy())):
        pred = worker.predict_frames(
            model_name, model, rgb, dep, torch.device("cpu"))
        assert pred.shape == (2, 4, 6)
        np.testing.assert_allclose(pred, 3.0)


def test_load_checkpoint_rejects_partial_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "best.pt"
    torch.save({"net": {"weight": model.weight.detach().clone()}}, path)
    try:
        worker.load_checkpoint_strict(model, path)
    except RuntimeError as error:
        assert "bias" in str(error)
    else:
        raise AssertionError("partial checkpoint was accepted")


def test_parser_requires_external_model_and_paths():
    parser = worker.make_parser()
    parsed = parser.parse_args([
        "--model", "dyspn",
        "--canonical-dir", "/input",
        "--checkpoint", "/weights/best.pt",
        "--args-json", "/weights/args.json",
        "--output", "/output/predictions.npz",
        "--device", "cuda:0",
    ])
    assert parsed.model == "dyspn"
