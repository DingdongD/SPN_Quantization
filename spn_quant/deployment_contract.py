"""Exact low-bit weight deployment contracts.

A deployment contract binds adaptive rounding to the exact folded graph used at
inference. It stores integer codes and scales, verifies the source graph, and
replays the learned weights without a second RTN pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn


FORMAT_VERSION = 1
SUPPORTED_MODULES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
    ).encode("utf-8")
    return header + b"\0" + value.numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _module_layout(module: nn.Module) -> Tuple[str, int]:
    if isinstance(module, nn.ConvTranspose2d):
        return "conv_transpose", int(module.groups)
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return "output_axis0", int(getattr(module, "groups", 1))
    raise TypeError(
        "unsupported deployment-contract module: %s" % type(module).__name__)


def _expand_scale(module: nn.Module, weight: torch.Tensor,
                  compact: torch.Tensor) -> torch.Tensor:
    layout, groups = _module_layout(module)
    compact = compact.to(device=weight.device, dtype=weight.dtype)
    if layout == "output_axis0" or groups == 1:
        return compact
    in_per_group = int(module.in_channels // groups)
    out_per_group = int(module.out_channels // groups)
    expanded = compact.expand(
        groups, in_per_group, out_per_group, *weight.shape[2:])
    return expanded.reshape_as(weight)


def output_channel_scales(module: nn.Module,
                          compact: torch.Tensor) -> torch.Tensor:
    compact = compact.detach().cpu().float()
    layout, groups = _module_layout(module)
    if layout == "output_axis0":
        return compact.reshape(compact.shape[0], -1)[:, 0]
    if groups == 1:
        return compact.reshape(-1)
    out_per_group = int(module.out_channels // groups)
    return compact.reshape(groups, -1, out_per_group).permute(
        0, 2, 1)[..., 0].reshape(-1)


def dequantize_weight_contract(module: nn.Module,
                               entry: Mapping[str, Any]) -> torch.Tensor:
    codes = torch.as_tensor(entry["codes"], dtype=torch.float32)
    scale = torch.as_tensor(entry["scale"], dtype=torch.float32)
    reference = module.weight.detach().cpu().float()
    if tuple(codes.shape) != tuple(reference.shape):
        raise ValueError(
            "weight-contract shape mismatch for %s" % entry.get("module", ""))
    expanded = _expand_scale(module, reference, scale)
    return codes * expanded


def export_rounding_contracts(controller: Any,
                              prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """Export exact codes/scales before an AdaRound controller is hardened."""
    output = {}
    for local_name in sorted(controller.parametrizations):
        parametrization = controller.parametrizations[local_name]
        module = controller.modules[local_name]
        original = module.parametrizations.weight.original.detach()
        expanded = parametrization._expanded_scale(original)
        codes = (
            torch.floor(original / expanded) + parametrization.hard_rounding()
        ).clamp(parametrization.qmin, parametrization.qmax).to(torch.int8)
        dequantized = codes.to(original.dtype) * expanded
        name = prefix if not local_name else (
            "%s.%s" % (prefix, local_name) if prefix else local_name)
        compact = parametrization.scale.detach().cpu().float().clone()
        layout, groups = _module_layout(module)
        bias = getattr(module, "bias", None)
        output[name] = {
            "module": name,
            "module_type": type(module).__name__,
            "bits": int(parametrization.config.bits),
            "qmin": int(parametrization.qmin),
            "qmax": int(parametrization.qmax),
            "layout": layout,
            "groups": int(groups),
            "shape": list(original.shape),
            "clip_ratio": float(parametrization.config.clip_ratio),
            "codes": codes.detach().cpu().clone(),
            "scale": compact,
            "output_scale": output_channel_scales(module, compact),
            "base_weight_sha256": tensor_sha256(original),
            "base_bias_sha256": tensor_sha256(bias) if bias is not None else "",
            "dequantized_sha256": tensor_sha256(dequantized),
            "code_sha256": tensor_sha256(codes),
        }
    return output


def build_deployment_contract(
        *, source_checkpoint: str | Path,
        graph_contract: Mapping[str, Any],
        weight_contracts: Mapping[str, Mapping[str, Any]],
        method: str, targets: Iterable[str],
        metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "method": str(method),
        "strict": 1,
        "source_checkpoint": str(Path(source_checkpoint)),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "targets": list(targets),
        "graph_contract": dict(graph_contract),
        "weight_contracts": dict(weight_contracts),
        "metadata": dict(metadata or {}),
    }


def save_deployment_contract(path: str | Path,
                             payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), str(path))
    return path


def load_deployment_contract(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(
            str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("deployment contract must be a dictionary")
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported deployment contract version")
    if not payload.get("strict"):
        raise ValueError("deployment contract is not marked strict")
    contracts = payload.get("weight_contracts")
    if not isinstance(contracts, dict) or not contracts:
        raise ValueError("deployment contract contains no weight contracts")
    return payload


def validate_graph_preparation(preparation: Mapping[str, Any],
                               graph_contract: Mapping[str, Any]) -> None:
    expected_folded = sorted(
        (row["conv"], row["bn"])
        for row in graph_contract.get("folded_pairs", []))
    actual_folded = sorted(
        (row["conv"], row["bn"])
        for row in preparation.get("folded_pairs", []))
    if actual_folded != expected_folded:
        raise RuntimeError(
            "deployment graph folding mismatch: expected=%s actual=%s" %
            (expected_folded, actual_folded))
    expected_unfolded = sorted(
        (row["conv"], row["bn"])
        for row in graph_contract.get("unfolded_fanout_pairs", []))
    actual_unfolded = sorted(
        (row["conv"], row["bn"])
        for row in preparation.get("unfolded_fanout_pairs", []))
    if actual_unfolded != expected_unfolded:
        raise RuntimeError("deployment graph excluded-fold mismatch")
    expected_plain = sorted(
        (row["conv"], row["bn"])
        for row in graph_contract.get("unfolded_conv_bn_pairs", []))
    actual_plain = sorted(
        (row["conv"], row["bn"])
        for row in preparation.get("unfolded_conv_bn_pairs", []))
    if actual_plain != expected_plain:
        raise RuntimeError("deployment graph fold-mode mismatch")
    expected_fold = bool(int(graph_contract.get("fold", 1)))
    if expected_fold and actual_plain:
        raise RuntimeError("deployment contract expects folded Conv-BN pairs")
    if not expected_fold and actual_folded:
        raise RuntimeError("deployment contract expects an unfolded graph")


class StrictContractInstrumentor(object):
    """Replay exact reconstructed weights after base activation-QDQ setup.

    The wrapped instrumentor still owns activation calibration and statistics.
    Contracted weights are overwritten with the exact exported code lattice
    after each configure call, so no second RTN pass can change them. Bias
    quantization remains owned by the active evaluation configuration.
    """

    def __init__(self, instrumentor: Any,
                 deployment_contract: Mapping[str, Any],
                 group_fn: Optional[Any] = None) -> None:
        self.instrumentor = instrumentor
        self.contract = dict(deployment_contract)
        self.entries = dict(self.contract["weight_contracts"])
        self.active_contracts = set()
        self.group_fn = group_fn
        self._register_missing_contract_modules()
        self._validate_base_model()

    def _register_missing_contract_modules(self) -> None:
        """Extend the legacy instrumentor for contracted ConvTranspose2d."""
        from scripts.hardware_aligned_quantization import HardwareMinMaxObserver

        modules = dict(self.instrumentor.model.named_modules())
        for name in sorted(self.entries):
            if name in self.instrumentor.original_weights:
                continue
            module = modules.get(name)
            if module is None or not isinstance(module, nn.ConvTranspose2d):
                continue
            if self.group_fn is None:
                raise RuntimeError(
                    "group_fn is required for contracted ConvTranspose2d: %s" %
                    name)
            group = self.group_fn(name, module)
            if group is None:
                raise RuntimeError(
                    "contracted ConvTranspose2d has no quantization group: %s" %
                    name)
            self.instrumentor.modules[name] = module
            self.instrumentor.groups[name] = str(group)
            self.instrumentor.original_weights[name] = (
                module.weight.detach().cpu().clone())
            self.instrumentor.original_biases[name] = (
                None if module.bias is None
                else module.bias.detach().cpu().clone())
            self.instrumentor.observers[(name, "input")] = (
                HardwareMinMaxObserver())
            self.instrumentor.observers[(name, "output")] = (
                HardwareMinMaxObserver())
            self.instrumentor.handles.append(
                module.register_forward_pre_hook(
                    self.instrumentor._make_pre_hook(name)))
            self.instrumentor.handles.append(
                module.register_forward_hook(
                    self.instrumentor._make_post_hook(name)))

    def _validate_base_model(self) -> None:
        modules = dict(self.instrumentor.model.named_modules())
        for name, entry in self.entries.items():
            module = modules.get(name)
            if module is None:
                raise KeyError(
                    "contracted module is missing after graph preparation: %s" %
                    name)
            if not isinstance(module, SUPPORTED_MODULES):
                raise TypeError("contracted module type changed: %s" % name)
            if name not in self.instrumentor.original_weights:
                raise RuntimeError(
                    "contracted module is not owned by the hardware "
                    "instrumentor: %s" % name)
            base_weight = self.instrumentor.original_weights[name]
            if tensor_sha256(base_weight) != entry["base_weight_sha256"]:
                raise RuntimeError(
                    "folded base weight fingerprint mismatch: %s" % name)
            base_bias = self.instrumentor.original_biases.get(name)
            expected_bias = entry.get("base_bias_sha256", "")
            actual_bias = tensor_sha256(base_bias) if base_bias is not None else ""
            if actual_bias != expected_bias:
                raise RuntimeError(
                    "folded base bias fingerprint mismatch: %s" % name)

    @staticmethod
    def _quantize_bias(bias: torch.Tensor, input_scale: float,
                       output_scale: torch.Tensor):
        scale = (
            output_scale.to(device=bias.device, dtype=bias.dtype) *
            float(input_scale))
        safe = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        limits = torch.iinfo(torch.int32)
        codes = torch.round(bias / safe).clamp(
            limits.min, limits.max).to(torch.int32)
        return codes.to(bias.dtype) * safe, codes, safe

    def configure(self, w_bits: int, a_bits: int,
                  enabled_groups: Iterable[str], **kwargs: Any) -> Any:
        enabled_groups = set(enabled_groups)
        active = {
            name for name in self.entries
            if self.instrumentor.groups.get(name) in enabled_groups
        }
        if active and any(
                int(self.entries[name]["bits"]) != int(w_bits)
                for name in active):
            raise ValueError(
                "deployment contract weight bits do not match configuration")
        if active and float(kwargs.get("weight_clip_ratio", 1.0)) != 1.0:
            raise ValueError(
                "weight clipping cannot be combined with an exact contract")
        smooth = kwargs.get("smooth_channel_maxima") or {}
        conflict = active & set(smooth)
        if conflict:
            raise ValueError(
                "SmoothQuant would invalidate exact weight contracts: %s" %
                sorted(conflict))

        self.instrumentor._restore_parameters()
        transpose_modules = {
            name: self.instrumentor.modules.pop(name)
            for name in sorted(active)
            if isinstance(
                self.instrumentor.modules.get(name), nn.ConvTranspose2d)
        }
        try:
            result = self.instrumentor.configure(
                w_bits, a_bits, enabled_groups, **kwargs)
        finally:
            self.instrumentor.modules.update(transpose_modules)
        self.active_contracts = active
        if transpose_modules:
            self._configure_transpose_activations(
                transpose_modules, a_bits,
                kwargs.get("activation_overrides") or {},
                kwargs.get("activation_bit_overrides") or {})
        self._apply_contracts(kwargs["quantize_bias"])
        return result

    def _configure_transpose_activations(
            self, modules: Mapping[str, nn.Module], a_bits: int,
            activation_overrides: Mapping[Any, Any],
            activation_bit_overrides: Mapping[Any, Any]) -> None:
        from scripts.hardware_aligned_quantization import (
            SymmetricActivationQuantizer,
            UnsignedActivationQuantizer,
        )
        from scripts.rtn_quantization import QuantizationStats

        for name in sorted(modules):
            for kind in ("input", "output"):
                key = (name, kind)
                observer = self.instrumentor.observers.get(key)
                if observer is None or not observer.observed:
                    raise RuntimeError(
                        "contracted ConvTranspose2d activation was not "
                        "observed: %s" % (key,))
                bits = int(activation_bit_overrides.get(
                    key, activation_bit_overrides.get(name, a_bits)))
                unsigned = kind == "input" and observer.minimum >= 0.0
                maximum = activation_overrides.get(key)
                if maximum is None:
                    quantizer = observer.quantizer(bits, unsigned=unsigned)
                elif unsigned:
                    quantizer = UnsignedActivationQuantizer(bits, maximum)
                else:
                    quantizer = SymmetricActivationQuantizer(bits, maximum)
                self.instrumentor.quantizers[key] = quantizer
                self.instrumentor.stats[key] = QuantizationStats()

    def _apply_contracts(self, quantize_bias: bool) -> None:
        from scripts.rtn_quantization import QuantizationStats

        modules = dict(self.instrumentor.model.named_modules())
        with torch.no_grad():
            for name in sorted(self.active_contracts):
                entry = self.entries[name]
                module = modules[name]
                dequantized = dequantize_weight_contract(module, entry)
                if tensor_sha256(dequantized) != entry["dequantized_sha256"]:
                    raise RuntimeError(
                        "weight-contract dequantized fingerprint mismatch: %s" %
                        name)
                module.weight.copy_(dequantized.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype))
                weight_stats = QuantizationStats()
                weight_stats.update(
                    self.instrumentor.original_weights[name], dequantized)
                weight_stats.exact_contract = 1
                self.instrumentor.stats[(name, "weight")] = weight_stats
                compact_scale = torch.as_tensor(entry["scale"]).float()
                self.instrumentor.weight_scales[name] = compact_scale

                original_bias = self.instrumentor.original_biases.get(name)
                if original_bias is not None and quantize_bias:
                    quantizer = self.instrumentor.quantizers.get((name, "input"))
                    if quantizer is None:
                        raise RuntimeError(
                            "contracted module has no input activation "
                            "quantizer: %s" % name)
                    output_scale = torch.as_tensor(
                        entry["output_scale"]).float()
                    quantized_bias, _, bias_scale = self._quantize_bias(
                        original_bias.to(module.bias.device),
                        quantizer.scale, output_scale)
                    module.bias.copy_(
                        quantized_bias.to(dtype=module.bias.dtype))
                    bias_stats = QuantizationStats()
                    bias_stats.update(
                        original_bias, quantized_bias.detach().cpu())
                    bias_stats.bias_scale = bias_scale.detach().cpu()
                    bias_stats.exact_contract = 1
                    self.instrumentor.stats[(name, "bias")] = bias_stats

    def manifest(self):
        rows = list(self.instrumentor.manifest())
        for name in sorted(self.active_contracts):
            entry = self.entries[name]
            rows.append({
                "module": name,
                "kind": "exact_weight_contract",
                "bits": int(entry["bits"]),
                "qmin": int(entry["qmin"]),
                "qmax": int(entry["qmax"]),
                "scale": "tensor:%s" % (tuple(
                    torch.as_tensor(entry["scale"]).shape),),
                "code_sha256": entry["code_sha256"],
                "dequantized_sha256": entry["dequantized_sha256"],
            })
        return rows

    def metadata(self):
        row = dict(self.instrumentor.metadata())
        row.update({
            "weight_execution": "exact_integer_code_contract",
            "contracted_weight_sites": len(self.active_contracts),
            "contract_format_version": int(
                self.contract["format_version"]),
        })
        return row

    def __getattr__(self, name: str) -> Any:
        return getattr(self.instrumentor, name)
