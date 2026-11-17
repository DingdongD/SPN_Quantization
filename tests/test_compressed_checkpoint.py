from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from spn_quant.compressed_checkpoint import (
    load_compressed_checkpoint,
    state_dict_sha256,
    verify_state_dict_sha256,
)


def _payload() -> dict:
    state = OrderedDict((
        ("weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        ("counter", torch.tensor(3, dtype=torch.int64)),
        ("bf16", torch.tensor([1.5], dtype=torch.bfloat16)),
    ))
    return {
        "format": "spn-independent-compressed-checkpoint",
        "format_version": 1,
        "model": "dyspn",
        "net": state,
        "net_sha256": state_dict_sha256(state),
        "architecture": {
            "nas_candidate": "drop_s4",
            "nas_depths": (3, 4, 6, 2),
            "structured_candidate": "bridge62_s5hidden62_s4hidden_75pct",
        },
        "quantization": {
            "candidate_id": "mixed",
            "assignment": {"weight_bits": {"encoder": 4}},
            "propagation_dtype": "bf16",
        },
        "model_build": {"saved_args": {"model": "dyspn"}},
        "calibration": {"indices": (1, 2)},
        "evidence": {"metric": "mean per-image RMSE"},
        "provenance": {"source_checkpoint_sha256": "abc"},
    }


def test_state_dict_hash_is_order_independent_and_tensor_sensitive():
    payload = _payload()
    state = payload["net"]
    reordered = OrderedDict(reversed(tuple(state.items())))
    altered = OrderedDict(state)
    altered["weight"] = state["weight"] + 1.0

    assert state_dict_sha256(state) == state_dict_sha256(reordered)
    assert state_dict_sha256(state) != state_dict_sha256(altered)
    assert verify_state_dict_sha256(state, payload["net_sha256"])
    assert not verify_state_dict_sha256(altered, payload["net_sha256"])


def test_load_compressed_checkpoint_validates_contract(tmp_path: Path):
    path = tmp_path / "dyspn.pt"
    torch.save(_payload(), path)

    checkpoint = load_compressed_checkpoint(path, expected_model="dyspn")

    assert checkpoint.model == "dyspn"
    assert checkpoint.architecture["nas_depths"] == (3, 4, 6, 2)
    assert checkpoint.quantization["propagation_dtype"] == "bf16"
    assert checkpoint.source_path == path.resolve()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(format="unknown"), "format"),
        (lambda payload: payload.update(format_version=2), "format"),
        (lambda payload: payload.update(model="nlspn"), "model identity"),
        (lambda payload: payload.pop("architecture"), "missing"),
    ],
)
def test_load_compressed_checkpoint_rejects_invalid_contract(
        tmp_path: Path, mutation, message: str):
    payload = _payload()
    mutation(payload)
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_compressed_checkpoint(path, expected_model="dyspn")


def test_load_compressed_checkpoint_rejects_altered_tensor(tmp_path: Path):
    payload = _payload()
    payload["net"]["weight"].add_(1.0)
    path = tmp_path / "altered.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="tensor hash"):
        load_compressed_checkpoint(path)
