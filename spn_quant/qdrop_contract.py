"""Versioned exact W4+A4 deployment contracts for QDrop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from spn_quant.deployment_contract import (
    StrictContractInstrumentor,
    file_sha256,
    tensor_sha256,
)
from spn_quant.qdrop_activation import ExactActivationQuantizer
from spn_quant.qdrop_targets import QDropActivationSite, QDropTargetPlan


QDROP_CONTRACT_VERSION = 2

QDROP_CONTRACT_FIELDS = {
    "format_version",
    "method",
    "strict",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "graph_contract",
    "weight_bits",
    "activation_bits",
    "activation_policy",
    "weight_contracts",
    "activation_contracts",
    "target_plan",
    "metadata",
    "fingerprint",
}

TARGET_PLAN_FIELDS = {
    "model", "blocks", "activation_sites", "excluded_sites"}

TARGET_SITE_FIELDS = {
    "site", "owner_name", "owner_kind", "role", "signed", "symmetric"}


def _target_plan_payload(plan):
    if not isinstance(plan, QDropTargetPlan):
        raise TypeError("QDrop contract targets must be QDropTargetPlan")
    return {
        "model": plan.model,
        "blocks": list(plan.blocks),
        "activation_sites": [
            {
                "site": site.site,
                "owner_name": site.owner_name,
                "owner_kind": site.owner_kind,
                "role": site.role,
                "signed": int(site.signed),
                "symmetric": int(site.symmetric),
            }
            for site in plan.activation_sites
        ],
        "excluded_sites": list(plan.excluded_sites),
    }


def _fingerprint_payload(payload):
    weight_rows = {}
    for name in sorted(payload["weight_contracts"]):
        entry = payload["weight_contracts"][name]
        weight_rows[name] = {
            "bits": int(entry["bits"]),
            "code_sha256": str(entry["code_sha256"]),
            "dequantized_sha256": str(entry["dequantized_sha256"]),
            "base_weight_sha256": str(entry["base_weight_sha256"]),
            "base_bias_sha256": str(entry["base_bias_sha256"]),
        }
    activation_rows = dict(
        (name, str(payload["activation_contracts"][name]["fingerprint"]))
        for name in sorted(payload["activation_contracts"]))
    return {
        "format_version": int(payload["format_version"]),
        "method": str(payload["method"]),
        "strict": int(payload["strict"]),
        "source_checkpoint_sha256": str(
            payload["source_checkpoint_sha256"]),
        "graph_contract": payload["graph_contract"],
        "weight_bits": int(payload["weight_bits"]),
        "activation_bits": int(payload["activation_bits"]),
        "activation_policy": str(payload["activation_policy"]),
        "weights": weight_rows,
        "activations": activation_rows,
        "target_plan": payload["target_plan"],
        "metadata": payload["metadata"],
    }


def _bundle_fingerprint(payload):
    encoded = json.dumps(
        _fingerprint_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_target_plan(payload):
    if set(payload) != TARGET_PLAN_FIELDS:
        raise KeyError("QDrop target plan fields mismatch")
    blocks = tuple(str(name) for name in payload["blocks"])
    if tuple(sorted(set(blocks))) != blocks or not blocks:
        raise ValueError("QDrop target blocks are invalid")
    sites = payload["activation_sites"]
    site_names = []
    for row in sites:
        if set(row) != TARGET_SITE_FIELDS:
            raise KeyError("QDrop target activation fields mismatch")
        if str(row["owner_name"]) not in blocks:
            raise ValueError("QDrop activation owner is not a target block")
        if bool(int(row["symmetric"])) and not bool(int(row["signed"])):
            raise ValueError("unsigned QDrop target cannot be symmetric")
        site_names.append(str(row["site"]))
    if len(set(site_names)) != len(site_names):
        raise ValueError("duplicate QDrop target activation site")
    if tuple(sorted(set(payload["excluded_sites"]))) != \
            tuple(payload["excluded_sites"]):
        raise ValueError("QDrop excluded target sites are invalid")
    return tuple(site_names)


def _validate_weight_contracts(contracts):
    if not isinstance(contracts, dict) or not contracts:
        raise ValueError("QDrop contract contains no W4 weights")
    for name, entry in contracts.items():
        if str(entry["module"]) != str(name):
            raise ValueError("QDrop weight module key mismatch")
        if int(entry["bits"]) != 4:
            raise ValueError("QDrop weight contract requires W4")
        codes = torch.as_tensor(entry["codes"])
        if tensor_sha256(codes) != str(entry["code_sha256"]):
            raise RuntimeError("QDrop weight code fingerprint mismatch: %s" %
                               name)


def _validate_activation_contracts(contracts):
    if not isinstance(contracts, dict) or not contracts:
        raise ValueError("QDrop contract contains no A4 activations")
    for name, entry in contracts.items():
        if str(entry["site"]) != str(name):
            raise ValueError("QDrop activation site key mismatch")
        ExactActivationQuantizer.from_contract(entry)


def _validate_qdrop_contract(payload):
    if not isinstance(payload, dict):
        raise TypeError("QDrop deployment contract must be a dictionary")
    if set(payload) != QDROP_CONTRACT_FIELDS:
        raise KeyError("QDrop deployment contract fields mismatch")
    if int(payload["format_version"]) != QDROP_CONTRACT_VERSION:
        raise ValueError("unsupported QDrop deployment contract version")
    if str(payload["method"]) != "qdrop_strict" or \
            int(payload["strict"]) != 1:
        raise ValueError("invalid strict QDrop deployment method")
    if int(payload["weight_bits"]) != 4 or \
            int(payload["activation_bits"]) != 4:
        raise ValueError("QDrop deployment contract requires W4A4")
    if str(payload["activation_policy"]) != \
            "exact_semantic_edge_contract":
        raise ValueError("invalid QDrop activation policy")
    _validate_weight_contracts(payload["weight_contracts"])
    _validate_activation_contracts(payload["activation_contracts"])
    target_sites = set(_validate_target_plan(payload["target_plan"]))
    contract_sites = set(payload["activation_contracts"])
    if target_sites != contract_sites:
        raise RuntimeError(
            "QDrop target and activation contract ownership mismatch")
    if _bundle_fingerprint(payload) != str(payload["fingerprint"]):
        raise RuntimeError("QDrop deployment contract fingerprint mismatch")
    return payload


def build_qdrop_contract(*, source_checkpoint, graph_contract,
                         weight_contracts, activation_contracts,
                         targets, metadata):
    checkpoint = Path(source_checkpoint)
    payload = {
        "format_version": QDROP_CONTRACT_VERSION,
        "method": "qdrop_strict",
        "strict": 1,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "graph_contract": dict(graph_contract),
        "weight_bits": 4,
        "activation_bits": 4,
        "activation_policy": "exact_semantic_edge_contract",
        "weight_contracts": dict(weight_contracts),
        "activation_contracts": dict(activation_contracts),
        "target_plan": _target_plan_payload(targets),
        "metadata": dict(metadata),
        "fingerprint": "",
    }
    payload["fingerprint"] = _bundle_fingerprint(payload)
    return _validate_qdrop_contract(payload)


def save_qdrop_contract(path, payload):
    _validate_qdrop_contract(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), output)
    return output


def load_qdrop_contract(path):
    payload = torch.load(
        Path(path), map_location="cpu", weights_only=False)
    return _validate_qdrop_contract(payload)


class QDropContractInstrumentor(object):
    def __init__(self, instrumentor, contract, group_fn=None):
        self.contract = _validate_qdrop_contract(dict(contract))
        weight_contract = {
            "format_version": 1,
            "weight_contracts": self.contract["weight_contracts"],
        }
        self.weight_instrumentor = StrictContractInstrumentor(
            instrumentor, weight_contract, group_fn=group_fn)
        self.instrumentor = self.weight_instrumentor.instrumentor
        self._site_rows = dict(
            (row["site"], row)
            for row in self.contract["target_plan"]["activation_sites"])
        self._quantizers = dict(
            (site, ExactActivationQuantizer.from_contract(entry))
            for site, entry in self.contract["activation_contracts"].items())
        self._active_sites = set()
        self._joint_adapter = None

    @property
    def model(self):
        return self.instrumentor.model

    @property
    def quantizers(self):
        return self.instrumentor.quantizers

    @property
    def relu_quantizers(self):
        return self.instrumentor.relu_quantizers

    @property
    def lognp_quantizers(self):
        return self.instrumentor.lognp_quantizers

    @property
    def lognp_relu_quantizers(self):
        return self.instrumentor.lognp_relu_quantizers

    @property
    def modules(self):
        return self.instrumentor.modules

    @property
    def observers(self):
        return self.instrumentor.observers

    @property
    def relu_observers(self):
        return self.instrumentor.relu_observers

    @property
    def activation_mode(self):
        return self.instrumentor.activation_mode

    @staticmethod
    def _module_boundary(row):
        parts = str(row["site"]).split("::")
        if len(parts) != 3 or parts[0] != "activation":
            raise ValueError("invalid QDrop module activation site")
        if str(row["role"]) == "module_input" and parts[2] == "input":
            return parts[1], "input"
        if str(row["role"]) == "module_output" and parts[2] == "output":
            return parts[1], "output"
        raise ValueError("QDrop module activation role mismatch")

    def _bind_generic_sites(self):
        for site in sorted(self._site_rows):
            row = self._site_rows[site]
            if row["owner_kind"] not in ("module_input", "module_output"):
                continue
            key = self._module_boundary(row)
            if key not in self.instrumentor.quantizers:
                raise KeyError("missing exact QDrop activation boundary: %s" %
                               (key,))
            self.instrumentor.quantizers[key] = self._quantizers[site]
            self._active_sites.add(site)

    def configure(self, w_bits, a_bits, enabled_groups, **kwargs):
        if int(w_bits) != 4 or int(a_bits) != 4:
            raise ValueError("exact QDrop replay requires W4A4")
        if str(kwargs["activation_mode"]) != "uniform":
            raise ValueError("exact QDrop replay requires uniform activation mode")
        if kwargs["activation_overrides"] or \
                kwargs["activation_bit_overrides"] or \
                kwargs["activation_format_overrides"] or \
                kwargs["smooth_channel_maxima"]:
            raise ValueError(
                "exact QDrop replay forbids activation overrides")
        if float(kwargs["weight_clip_ratio"]) != 1.0:
            raise ValueError("exact QDrop replay forbids weight clipping")
        quantize_bias = bool(kwargs["quantize_bias"])
        base_kwargs = dict(kwargs)
        base_kwargs["quantize_bias"] = False
        result = self.weight_instrumentor.configure(
            w_bits, a_bits, enabled_groups, **base_kwargs)
        if self.weight_instrumentor.active_contracts != \
                set(self.contract["weight_contracts"]):
            raise RuntimeError(
                "exact QDrop replay requires every contracted W4 weight")
        self._active_sites = set()
        self._bind_generic_sites()
        if quantize_bias:
            self.weight_instrumentor._apply_contracts(True)
        return result

    def bind_joint_adapter(self, adapter):
        if self._joint_adapter is not None:
            raise RuntimeError("QDrop joint adapter is already bound")
        sites = []
        quantizers = {}
        for site_name in sorted(self._site_rows):
            row = self._site_rows[site_name]
            if row["owner_kind"] not in ("attention_qkv", "concat_input"):
                continue
            site = QDropActivationSite(
                site=str(row["site"]),
                owner_name=str(row["owner_name"]),
                owner_kind=str(row["owner_kind"]),
                role=str(row["role"]),
                signed=bool(int(row["signed"])),
                symmetric=bool(int(row["symmetric"])),
            )
            sites.append(site)
            quantizers[site.site] = self._quantizers[site.site]
        if sites:
            adapter.bind_qdrop_sites(tuple(sites), quantizers)
            self._active_sites.update(quantizers)
        self._joint_adapter = adapter

    def observe(self, activation_mode="uniform"):
        return self.instrumentor.observe(activation_mode=activation_mode)

    def freeze(self):
        return self.instrumentor.freeze()

    def disable(self):
        return self.instrumentor.disable()

    def close(self):
        if self._joint_adapter is not None:
            self._joint_adapter.unbind_qdrop_sites()
        return self.instrumentor.close()

    def manifest(self):
        rows = list(self.weight_instrumentor.manifest())
        for site in sorted(self._active_sites):
            entry = self.contract["activation_contracts"][site]
            row = self._site_rows[site]
            rows.append({
                "module": site,
                "kind": "exact_activation_contract",
                "bits": int(entry["bits"]),
                "qmin": int(entry["qmin"]),
                "qmax": int(entry["qmax"]),
                "scale": float(torch.as_tensor(entry["scale"]).item()),
                "zero_point": int(entry["zero_point"]),
                "owner_name": str(row["owner_name"]),
                "owner_kind": str(row["owner_kind"]),
                "fingerprint": str(entry["fingerprint"]),
            })
        return rows

    def metadata(self):
        row = dict(self.weight_instrumentor.metadata())
        row.update({
            "activation_execution": "exact_integer_code_contract",
            "contracted_activation_sites": len(self._active_sites),
            "pending_activation_sites": len(self._quantizers) -
            len(self._active_sites),
            "qdrop_contract_format_version": QDROP_CONTRACT_VERSION,
        })
        return row

    def module_groups(self):
        return self.instrumentor.module_groups()

    def externally_owned_outputs(self):
        return self.instrumentor.externally_owned_outputs()

    def set_external_ownership(self, inputs, outputs):
        return self.instrumentor.set_external_ownership(inputs, outputs)

    def enable_compensation_capture(self, modules=None, sample_limit=8192):
        return self.instrumentor.enable_compensation_capture(
            modules=modules, sample_limit=sample_limit)

    def apply_compensation(self):
        return self.instrumentor.apply_compensation()

    def statistics(self):
        return self.instrumentor.statistics()

    def weight_bits_by_module(self):
        return self.instrumentor.weight_bits_by_module()

    def layernorm_fusions(self):
        return self.instrumentor.layernorm_fusions()

    def per_channel_activation_modules(self):
        return self.instrumentor.per_channel_activation_modules()
