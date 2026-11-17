"""Validation helpers for standalone NAS/pruned/quantized SPN checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping

import torch


FORMAT_NAME = "spn-independent-compressed-checkpoint"
FORMAT_VERSION = 1
SUPPORTED_MODELS = ("cspn", "dyspn", "nlspn", "completionformer")
_REQUIRED_KEYS = (
    "format",
    "format_version",
    "model",
    "net",
    "net_sha256",
    "architecture",
    "quantization",
    "model_build",
    "calibration",
    "evidence",
    "provenance",
)


@dataclass(frozen=True)
class CompressedCheckpoint:
    """A validated independent compressed-model payload."""

    source_path: Path
    model: str
    net: Mapping[str, torch.Tensor]
    net_sha256: str
    architecture: Mapping[str, Any]
    quantization: Mapping[str, Any]
    model_build: Mapping[str, Any]
    calibration: Mapping[str, Any]
    evidence: Mapping[str, Any]
    provenance: Mapping[str, Any]


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, types, shapes, and values independent of key order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError("checkpoint state values must be tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            if tensor.dtype != torch.bfloat16:
                raise
            raw = tensor.float().numpy().tobytes(order="C")
        digest.update(raw)
    return digest.hexdigest()


def verify_state_dict_sha256(
        state: Mapping[str, torch.Tensor], expected: str) -> bool:
    """Return whether a state dictionary matches its declared SHA-256."""
    return hmac.compare_digest(state_dict_sha256(state), str(expected))


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ValueError("compressed checkpoint %s must be a mapping" % key)
    return value


def load_compressed_checkpoint(
        path: Path, expected_model: str | None = None,
        verify: bool = True) -> CompressedCheckpoint:
    """Load and validate an independent compressed SPN checkpoint."""
    source = Path(path)
    payload = torch.load(str(source), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("compressed checkpoint payload must be a mapping")
    missing = tuple(key for key in _REQUIRED_KEYS if key not in payload)
    if missing:
        raise ValueError("compressed checkpoint is missing fields: %s" %
                         (missing,))
    if payload["format"] != FORMAT_NAME or \
            int(payload["format_version"]) != FORMAT_VERSION:
        raise ValueError("unsupported compressed checkpoint format")

    model = str(payload["model"])
    if model not in SUPPORTED_MODELS:
        raise ValueError("unsupported compressed checkpoint model: %s" % model)
    if expected_model is not None and model != str(expected_model):
        raise ValueError("compressed checkpoint model identity mismatch")

    state = _require_mapping(payload, "net")
    if not state:
        raise ValueError("compressed checkpoint state is empty")
    architecture = _require_mapping(payload, "architecture")
    quantization = _require_mapping(payload, "quantization")
    model_build = _require_mapping(payload, "model_build")
    calibration = _require_mapping(payload, "calibration")
    evidence = _require_mapping(payload, "evidence")
    provenance = _require_mapping(payload, "provenance")
    for key in ("nas_candidate", "nas_depths", "structured_candidate"):
        if key not in architecture:
            raise ValueError(
                "compressed checkpoint architecture is missing %s" % key)
    if quantization.get("propagation_dtype") not in ("fp16", "bf16"):
        raise ValueError("compressed checkpoint propagation dtype is invalid")
    net_hash = str(payload["net_sha256"])
    if verify and not verify_state_dict_sha256(state, net_hash):
        raise ValueError("compressed checkpoint tensor hash mismatch")

    return CompressedCheckpoint(
        source_path=source.resolve(),
        model=model,
        net=state,
        net_sha256=net_hash,
        architecture=architecture,
        quantization=quantization,
        model_build=model_build,
        calibration=calibration,
        evidence=evidence,
        provenance=provenance,
    )
