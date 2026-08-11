"""Bind learnable QDrop quantizers to deployment activation boundaries."""

from __future__ import annotations

import torch

from spn_quant.qdrop_activation import QDropActivationQuantizer


class QDropActivationBank(object):
    def __init__(self, plan, instrumentor, bits, scale_minimum, seed,
                 joint_adapter=None):
        self.plan = plan
        self.instrumentor = instrumentor
        self.bits = int(bits)
        self.scale_minimum = float(scale_minimum)
        self.seed = int(seed)
        self.joint_adapter = joint_adapter
        if self.bits != 4:
            raise ValueError("QDrop activation bank requires A4")
        self.phase = "created"
        self.quantizers = {}
        self._sites = dict(
            (site.site, site) for site in self.plan.activation_sites)
        self._snapshots = {}
        self._bound_targets = set()

    @staticmethod
    def _module_boundary(site):
        parts = site.site.split("::")
        if len(parts) != 3 or parts[0] != "activation":
            raise ValueError("invalid QDrop module boundary: %s" % site.site)
        if site.role == "module_input" and parts[2] == "input":
            return parts[1], "input"
        if site.role == "module_output" and parts[2] == "output":
            return parts[1], "output"
        raise ValueError("QDrop module boundary role mismatch: %s" % site.site)

    @staticmethod
    def _observer_tensor(observer):
        if not observer.observed:
            raise RuntimeError("QDrop activation boundary was not observed")
        minimum = torch.as_tensor(observer.minimum).detach().float().reshape(-1)
        maximum = torch.as_tensor(observer.maximum).detach().float().reshape(-1)
        return torch.cat((minimum, maximum))

    def observe(self):
        if self.phase != "created":
            raise RuntimeError("QDrop observation phase is already closed")
        self.instrumentor.observe(activation_mode="uniform")
        self.phase = "observing"

    def _generic_initialization(self, site):
        name, kind = self._module_boundary(site)
        key = (name, kind)
        if key not in self.instrumentor.quantizers:
            raise KeyError("missing QDrop hardware boundary: %s" % (key,))
        if key not in self.instrumentor.observers:
            raise KeyError("missing QDrop activation observer: %s" % (key,))
        if name not in self.instrumentor.modules:
            raise KeyError("missing QDrop hardware module: %s" % name)
        tensor = self._observer_tensor(self.instrumentor.observers[key])
        device = self.instrumentor.modules[name].weight.device
        return tensor.to(device=device)

    def _initialization_tensor(self, site):
        if site.owner_kind in ("module_input", "module_output"):
            return self._generic_initialization(site)
        if site.owner_kind in ("attention_qkv", "concat_input"):
            if self.joint_adapter is None:
                raise RuntimeError(
                    "CompletionFormer QDrop sites require the joint adapter")
            return self.joint_adapter.qdrop_initialization_tensor(site)
        raise ValueError("unsupported QDrop activation owner: %s" %
                         site.owner_kind)

    def _suppress_duplicate_boundaries(self):
        retained = set()
        for site in self.plan.activation_sites:
            if site.owner_kind in ("module_input", "module_output"):
                retained.add(self._module_boundary(site))
        for key in tuple(self.instrumentor.quantizers):
            if key in retained:
                continue
            label = "suppressed::activation::%s::%s" % key
            self._snapshots[label] = (
                self.instrumentor.quantizers,
                key,
                self.instrumentor.quantizers[key])
            del self.instrumentor.quantizers[key]
        for key in tuple(self.instrumentor.relu_quantizers):
            label = "suppressed::relu::%s" % key
            self._snapshots[label] = (
                self.instrumentor.relu_quantizers,
                key,
                self.instrumentor.relu_quantizers[key])
            del self.instrumentor.relu_quantizers[key]

    def initialize(self):
        if self.phase not in ("created", "observing"):
            raise RuntimeError("QDrop activation bank is already initialized")
        if not self.instrumentor.frozen:
            raise RuntimeError("hardware activation calibration is not frozen")
        if self.instrumentor.mode != "quantize" or \
                int(self.instrumentor.a_bits) != self.bits or \
                self.instrumentor.activation_mode != "uniform":
            raise RuntimeError(
                "QDrop requires configured uniform A4 hardware boundaries")
        for index, site in enumerate(self.plan.activation_sites):
            tensor = self._initialization_tensor(site)
            quantizer = QDropActivationQuantizer(
                site=site.site,
                bits=self.bits,
                signed=site.signed,
                symmetric=site.symmetric,
                scale_minimum=self.scale_minimum,
                seed=self.seed + index,
            ).to(device=tensor.device)
            quantizer.initialize(tensor)
            self.quantizers[site.site] = quantizer
        self._suppress_duplicate_boundaries()
        self.phase = "initialized"

    def _bind_generic(self, site):
        name, kind = self._module_boundary(site)
        key = (name, kind)
        if key not in self.instrumentor.quantizers:
            raise KeyError("missing QDrop hardware boundary: %s" % (key,))
        if site.site in self._snapshots:
            raise RuntimeError("QDrop hardware boundary is already replaced")
        original = self.instrumentor.quantizers[key]
        if any(
                snapshot[0] is self.instrumentor.quantizers and
                snapshot[1] == key
                for snapshot in self._snapshots.values()):
            raise RuntimeError("QDrop hardware boundary has multiple owners")
        self._snapshots[site.site] = (
            self.instrumentor.quantizers, key, original)
        self.instrumentor.quantizers[key] = self.quantizers[site.site]

    def _bind_target(self, target):
        sites = tuple(
            site for site in self.plan.activation_sites
            if site.owner_name == target)
        if not sites:
            self._bound_targets.add(target)
            return
        joint_sites = []
        joint_quantizers = {}
        for site in sites:
            if site.owner_kind in ("module_input", "module_output"):
                self._bind_generic(site)
            else:
                joint_sites.append(site)
                joint_quantizers[site.site] = self.quantizers[site.site]
        if joint_sites:
            if self.joint_adapter is None:
                raise RuntimeError(
                    "CompletionFormer QDrop sites require the joint adapter")
            self.joint_adapter.bind_qdrop_sites(
                tuple(joint_sites), joint_quantizers)
        self._bound_targets.add(target)

    def reconstruct(self, target, quant_probability):
        if self.phase not in ("initialized", "reconstruction", "partially_frozen"):
            raise RuntimeError("QDrop activation bank is not initialized")
        if target not in self.plan.blocks:
            raise KeyError("unknown QDrop target: %s" % target)
        if target in self._bound_targets:
            raise RuntimeError("QDrop target is already bound: %s" % target)
        self._bind_target(target)
        for site in self.plan.activation_sites:
            if site.owner_name == target:
                self.quantizers[site.site].start_reconstruction(
                    quant_probability)
        self.phase = "reconstruction"

    def parameters_for(self, target):
        if target not in self.plan.blocks:
            raise KeyError("unknown QDrop target: %s" % target)
        parameters = []
        for site in self.plan.activation_sites:
            if site.owner_name == target:
                parameters.extend(self.quantizers[site.site].parameters())
        return tuple(parameters)

    def set_quant_probability(self, target, quant_probability):
        if target not in self._bound_targets:
            raise RuntimeError("QDrop target is not bound: %s" % target)
        for site in self.plan.activation_sites:
            if site.owner_name == target:
                self.quantizers[site.site].set_quant_probability(
                    quant_probability)

    def freeze_target(self, target):
        if target not in self._bound_targets:
            raise RuntimeError("QDrop target is not bound: %s" % target)
        joint = False
        for site in self.plan.activation_sites:
            if site.owner_name != target:
                continue
            if site.owner_kind in ("attention_qkv", "concat_input"):
                joint = True
            else:
                self.quantizers[site.site].freeze()
        if joint:
            self.joint_adapter.freeze_qdrop_sites(target)
        self.phase = "partially_frozen"

    def disable_randomness(self):
        incomplete = sorted(
            site for site, quantizer in self.quantizers.items()
            if quantizer.phase != "frozen")
        if incomplete:
            raise RuntimeError(
                "QDrop activation sites are not frozen: %s" % incomplete)
        self.phase = "frozen"

    def contracts(self):
        contracts = {}
        for site, quantizer in self.quantizers.items():
            if quantizer.phase == "frozen":
                contracts[site] = quantizer.contract()
        return contracts

    def manifest(self):
        rows = []
        for site in self.plan.activation_sites:
            quantizer = self.quantizers[site.site]
            row = quantizer.statistics()
            row.update({
                "owner_name": site.owner_name,
                "owner_kind": site.owner_kind,
                "role": site.role,
                "phase": quantizer.phase,
            })
            rows.append(row)
        return rows

    def close(self):
        if self.phase == "closed":
            return
        if self.joint_adapter is not None and self._bound_targets:
            self.joint_adapter.unbind_qdrop_sites()
        for container, key, original in self._snapshots.values():
            container[key] = original
        self._snapshots = {}
        self._bound_targets = set()
        self.phase = "closed"
