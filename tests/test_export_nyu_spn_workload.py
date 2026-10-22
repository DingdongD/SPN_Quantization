from scripts.export_nyu_spn_workload import (
    neighborhood_size,
    runtime_args,
    select_sample_index,
)


def _model_spec():
    return {
        "model": "nlspn",
        "run_dir": "/runs/nlspn",
        "checkpoint": "/runs/nlspn/best.pt",
        "expected_architecture_class": "NLSPNModel",
        "required_cuda_extension": "DCN",
        "propagation_iterations": 18,
        "data_root": "/data/nyu",
        "checkpoint_architecture": "NLSPN resnet34",
        "native_cuda_operator": None,
        "evaluation_indices": [386, 572],
    }


def test_exporter_uses_first_fixed_evaluation_identity():
    assert select_sample_index(_model_spec()) == 386


def test_runtime_args_preserve_model_identity():
    args = runtime_args(_model_spec(), "cuda:2")
    assert args.model == "nlspn"
    assert args.device == "cuda:2"
    assert args.propagation_iterations == 18
    assert str(args.checkpoint) == "/runs/nlspn/best.pt"


class _ModelWithNeighbors(object):
    num_neighbors = 8


class _DyBase(object):
    num_sample = 5


class _DyModel(object):
    base = _DyBase()


def test_neighborhood_size_is_model_specific():
    assert neighborhood_size("cspn", object()) == 8
    assert neighborhood_size("dyspn", _DyModel()) == 5
    assert neighborhood_size("nlspn", _ModelWithNeighbors()) == 8
    assert neighborhood_size("completionformer", _ModelWithNeighbors()) == 8
