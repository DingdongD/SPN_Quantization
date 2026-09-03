"""Shared model-level semantic quantization adapter primitives."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.runtime import EdgeQDQRuntime
from spn_quant.sites import QuantSite, QuantSiteRegistry
from spn_quant.specs import QuantSpec


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _unsigned(bits: int, observer: str = "minmax",
              dynamic: bool = False) -> QuantSpec:
    return QuantSpec(bits=bits, scheme="affine", granularity="tensor",
                     observer=observer, transform="none", signed=False,
                     preserve_zero=True, dynamic=dynamic)


def semantic_spec(role: str) -> QuantSpec:
    if role == "rgb_input":
        return _unsigned(8)
    if role == "sparse_mask":
        return _unsigned(1)
    if role == "sparse_depth_value":
        return _unsigned(4, "zero_aware")
    if role in ("initial_depth", "confidence", "prediction"):
        return _unsigned(4)
    if role == "propagation_state":
        return _unsigned(4, dynamic=True)
    return QuantSpec.signed_tensor(4)


def recommended_bits(role: str) -> int:
    if role == "sparse_mask":
        return 1
    if role in {
        "rgb_input", "guidance_logits", "affinity_logits", "affinity",
        "confidence_logits", "confidence", "offset_logits", "offset",
        "propagation_state", "initial_depth", "prediction", "se_gate",
        "channel_attention_gate", "spatial_attention_gate",
        "attention_probability",
    }:
        return 8
    return 4


@dataclass(frozen=True)
class ModuleRoleRule:
    pattern: str
    role: str
    required: bool = False
    priority: int = 0

    def matches(self, name: str) -> bool:
        return re.search(self.pattern, name) is not None


@dataclass(frozen=True)
class SignalRule:
    name: str
    role: str
    source: str
    key: str
    producer: str
    required: bool = True


@dataclass(frozen=True)
class PatternSignalRule:
    pattern: str
    role: str
    required: bool = False
    prefix: str = "signal"

    def matches(self, name: str) -> bool:
        return re.search(self.pattern, name) is not None


class ModelSemanticAdapter:
    """Register, observe and validate semantic quantization sites."""
    MODEL_NAME = "base"
    MODULE_RULES: Tuple[ModuleRoleRule, ...] = ()
    SIGNAL_RULES: Tuple[SignalRule, ...] = ()
    PATTERN_SIGNAL_RULES: Tuple[PatternSignalRule, ...] = ()
    REQUIRED_ROLES: Tuple[str, ...] = ()
    PROPAGATION_PATHS: Tuple[str, ...] = ()
    ALLOWED_CONCAT_CALLS: Optional[Tuple[int, ...]] = None
    CONTRACT_PROTECTED_ROLES: Tuple[str, ...] = ()
    CONTRACT_PREFIX_GROUP_PATTERNS: Tuple[Tuple[str, ...], ...] = ()
    CONTRACT_TAIL_GROUP_PATTERNS: Tuple[Tuple[str, ...], ...] = ()

    @classmethod
    def module_manifest(cls, model: nn.Module) -> Tuple[Dict[str, Any], ...]:
        """Return semantic roles for every supported weight module."""
        quant_types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        rows = []
        for name, module in model.named_modules():
            if not name or not isinstance(module, quant_types):
                continue
            rule = cls._module_rule(name)
            rows.append({
                "name": name,
                "module": module,
                "role": "" if rule is None else rule.role,
            })
        return tuple(rows)

    @classmethod
    def _module_rule(cls, name: str) -> Optional[ModuleRoleRule]:
        matches = [rule for rule in cls.MODULE_RULES if rule.matches(name)]
        return sorted(matches, key=lambda rule: (-rule.priority, rule.pattern))[0] \
            if matches else None

    def __init__(self, model: nn.Module,
                 runtime: Optional[EdgeQDQRuntime] = None,
                 merge_policy: str = "shared",
                 group_size: Optional[int] = None,
                 strict: bool = True) -> None:
        self.model = model
        self.runtime = runtime or EdgeQDQRuntime()
        self.merge_policy = merge_policy
        self.group_size = group_size
        self.strict = strict
        self.mode = "bypass"
        self.registry = QuantSiteRegistry()
        self._handles: List[Any] = []
        self._observations: Dict[str, Dict[str, Any]] = {}
        self._module_sites: Dict[str, str] = {}
        self._pattern_sites: Dict[str, str] = {}
        self._task_capture_enabled = False
        self._task_capture_forwards = 0
        self._task_capture_values: Dict[str, Any] = {}
        self._task_signal_values: Dict[str, Any] = {}
        self._register_inputs()
        self._register_modules()
        self._register_signals()
        self._register_pattern_signals()
        self._register_declared_sites()
        self._install_capture_hooks()
        self._merge_adapters = list(self._build_merge_adapters())
        self._quantization_delegated = False

    def _register(self, name: str, role: str, producer: str,
                  required: bool, source: str,
                  metadata: Optional[Mapping[str, Any]] = None) -> None:
        meta = {
            "model": self.MODEL_NAME, "source": source,
            "required": int(required), "recommended_bits": recommended_bits(role),
            "operational": 1,
        }
        meta.update(dict(metadata or {}))
        self.registry.register(QuantSite(
            name=name, role=role, producer=producer, consumers=(),
            spec=semantic_spec(role), metadata=meta))

    def _register_inputs(self) -> None:
        for name, role in (("rgb", "rgb_input"),
                           ("sparse_depth", "sparse_depth_value"),
                           ("sparse_mask", "sparse_mask")):
            self._register("input::" + name, role, "model_input", True,
                           "model_input")

    def _rule(self, name: str) -> Optional[ModuleRoleRule]:
        return self._module_rule(name)

    def _register_modules(self) -> None:
        quant_types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        for name, module in self.model.named_modules():
            if not name or not isinstance(module, quant_types):
                continue
            rule = self._rule(name)
            if rule is None:
                continue
            site = "activation::" + name
            self._register(site, rule.role, name, rule.required,
                           "module_output",
                           {"module_class": module.__class__.__name__})
            self._module_sites[name] = site

    def _register_signals(self) -> None:
        for rule in self.SIGNAL_RULES:
            self._register(rule.name, rule.role, rule.producer, rule.required,
                           "semantic_signal",
                           {"capture_source": rule.source,
                            "capture_key": rule.key})

    def _register_pattern_signals(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                continue
            for rule in self.PATTERN_SIGNAL_RULES:
                if rule.matches(name):
                    site = "%s::%s" % (rule.prefix, name)
                    self._register(site, rule.role, name, rule.required,
                                   "pattern_signal",
                                   {"module_class": module.__class__.__name__,
                                    "pattern": rule.pattern})
                    self._pattern_sites[name] = site
                    break

    def declared_merge_sites(self) -> Sequence[Tuple[str, str, bool]]:
        return ()

    def _register_declared_sites(self) -> None:
        for name, role, operational in self.declared_merge_sites():
            self._register(name, role, name, False, "declared_site",
                           {"operational": int(operational)})

    def _record(self, site: str, value: Any) -> None:
        if self.mode != "observe" or site not in self.registry:
            return
        tensors = list(_iter_tensors(value))
        if not tensors:
            return
        if self._task_capture_enabled:
            for tensor in tensors:
                if tensor.requires_grad:
                    tensor.retain_grad()
            self._task_signal_values[site] = value if len(tensors) != 1 \
                else tensors[0]
        row = self._observations.setdefault(site, {
            "observations": 0, "numel": 0, "nonfinite": 0,
            "shapes": set(), "dtypes": set(),
        })
        row["observations"] += 1
        for tensor in tensors:
            row["numel"] += tensor.numel()
            row["shapes"].add("x".join(map(str, tensor.shape)))
            row["dtypes"].add(str(tensor.dtype).replace("torch.", ""))
            if tensor.is_floating_point():
                row["nonfinite"] += int((~torch.isfinite(tensor.detach())).sum())

    def _input_signals(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        raise NotImplementedError

    def _propagation_inputs(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        return {}

    def _propagation_outputs(self, output: Any) -> Mapping[str, Any]:
        return {}

    def _model_outputs(self, output: Any) -> Mapping[str, Any]:
        if torch.is_tensor(output):
            return {"prediction": output}
        if not isinstance(output, Mapping):
            return {}
        states = output.get("pred_inter")
        if states is None:
            states = output.get("list_feat")
        return {
            "prediction": output.get("pred"),
            "initial_depth": output.get("pred_init"),
            "guidance": output.get("guidance"),
            "confidence": output.get("confidence"),
            "offset": output.get("offset"),
            "affinity": output.get("aff"),
            "propagation_state": states,
        }

    def _resolve_propagation_module(self) -> Optional[nn.Module]:
        modules = dict(self.model.named_modules())
        for name in self.PROPAGATION_PATHS:
            if name in modules:
                return modules[name]
        return None

    def _propagation_module(self) -> Optional[nn.Module]:
        return self._resolve_propagation_module()

    def _record_rules(self, source: str, values: Mapping[str, Any]) -> None:
        for rule in self.SIGNAL_RULES:
            if rule.source == source and values.get(rule.key) is not None:
                self._record(rule.name, values[rule.key])

    def _capture_task_values(self, source: str,
                             values: Mapping[str, Any]) -> None:
        if not self._task_capture_enabled:
            return
        if source == "prop_input" and \
                values.get("initial_depth") is not None:
            self._task_capture_values["initial_depth"] = \
                values["initial_depth"]
        elif source == "prop_output" and \
                values.get("propagation_state") is not None:
            self._task_capture_values["propagation_state"] = \
                tuple(_iter_tensors(values["propagation_state"]))
        elif source == "model_output" and \
                values.get("prediction") is not None:
            self._task_capture_values["prediction"] = values["prediction"]
            self._task_capture_forwards += 1

    def _install_capture_hooks(self) -> None:
        modules = dict(self.model.named_modules())

        def root_pre(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            del module
            if self.mode != "observe":
                return
            values = self._input_signals(inputs)
            for key, value in values.items():
                self._record("input::" + key, value)
            sparse = values.get("sparse_depth")
            if torch.is_tensor(sparse):
                self._record("input::sparse_mask", sparse.ne(0))

        def root_post(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            del module, inputs
            values = self._model_outputs(output)
            self._record_rules("model_output", values)
            self._capture_task_values("model_output", values)

        self._handles += [self.model.register_forward_pre_hook(root_pre),
                          self.model.register_forward_hook(root_post)]

        for name, site in {**self._module_sites,
                           **self._pattern_sites}.items():
            def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any,
                     target: str = site) -> None:
                del module, inputs
                self._record(target, output)
            self._handles.append(modules[name].register_forward_hook(hook))

        propagation = self._propagation_module()
        if propagation is None:
            if self.strict:
                raise RuntimeError("%s propagation module was not found" %
                                   self.MODEL_NAME)
        else:
            def prop_pre(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
                del module
                values = self._propagation_inputs(inputs)
                self._record_rules("prop_input", values)
                self._capture_task_values("prop_input", values)
            def prop_post(module: nn.Module, inputs: Tuple[Any, ...],
                          output: Any) -> None:
                del module, inputs
                values = self._propagation_outputs(output)
                self._record_rules("prop_output", values)
                self._capture_task_values("prop_output", values)
            self._handles += [propagation.register_forward_pre_hook(prop_pre),
                              propagation.register_forward_hook(prop_post)]

        for rule in self.SIGNAL_RULES:
            if rule.source != "module_output":
                continue
            module = modules.get(rule.producer)
            if module is None:
                if rule.required and self.strict:
                    raise RuntimeError("required semantic producer missing: %s" %
                                       rule.producer)
                continue
            def signal_hook(current: nn.Module, inputs: Tuple[Any, ...],
                            output: Any, target: str = rule.name) -> None:
                del current, inputs
                self._record(target, output)
            self._handles.append(module.register_forward_hook(signal_hook))
        self._install_extra_hooks()

    def _install_extra_hooks(self) -> None:
        pass

    def _build_merge_adapters(self) -> Sequence[Any]:
        from scripts.hardware_merge_adapters import CallIndexedConcatAdapter
        return (CallIndexedConcatAdapter(
            self.model, policy=self.merge_policy, group_size=self.group_size,
            runtime=self.runtime, manage_runtime=False),)

    def observe(self) -> None:
        self.mode = "observe"
        self._observations = {}
        for adapter in self._merge_adapters:
            adapter.observe()

    def begin_task_capture(self) -> None:
        """Start one differentiable semantic capture for the next forward."""
        self._task_capture_enabled = True
        self._task_capture_forwards = 0
        self._task_capture_values = {}
        self._task_signal_values = {}

    def task_signal_values(self):
        """Return live semantic tensors captured by the next forward."""
        if self._task_capture_enabled:
            raise RuntimeError("semantic task capture is still active")
        if not self._task_signal_values:
            raise RuntimeError("semantic task signal capture is empty")
        return dict(self._task_signal_values)

    def task_capture(self):
        """Return initial depth and propagation tensors normalized by adapter."""
        from spn_quant.qat.task_loss import ModelTaskCapture
        if not self._task_capture_enabled:
            raise RuntimeError("semantic task capture is not enabled")
        if self._task_capture_forwards != 1:
            raise RuntimeError(
                "semantic task capture requires exactly one model forward")
        required = {"prediction", "initial_depth", "propagation_state"}
        if set(self._task_capture_values) != required:
            raise RuntimeError(
                "semantic task capture is incomplete: %s" % sorted(
                    required - set(self._task_capture_values)))
        prediction = self._task_capture_values["prediction"]
        initial_depth = self._task_capture_values["initial_depth"]
        states = self._task_capture_values["propagation_state"]
        if not torch.is_tensor(prediction) or not torch.is_tensor(initial_depth):
            raise TypeError("semantic depth captures must be tensors")
        if not states or any(not torch.is_tensor(state) for state in states):
            raise TypeError("semantic propagation captures must be tensors")
        capture = ModelTaskCapture(
            prediction=prediction,
            initial_depth=initial_depth,
            propagation_states=tuple(states),
        )
        self._task_capture_enabled = False
        return capture

    def delegate_merge_quantization(self) -> None:
        if self.mode != "bypass":
            raise RuntimeError(
                "merge quantization ownership must be delegated before observation")
        for adapter in self._merge_adapters:
            adapter.close()
        self._merge_adapters = []
        self.ALLOWED_CONCAT_CALLS = (0,)

    def delegate_quantization(self) -> None:
        self.delegate_merge_quantization()
        self._quantization_delegated = True

    def _validate(self) -> None:
        observed_roles = {site.role for site in self.registry
                          if site.name in self._observations}
        missing_roles = sorted(set(self.REQUIRED_ROLES) - observed_roles)
        if missing_roles:
            raise RuntimeError("%s semantic roles were not observed: %s" %
                               (self.MODEL_NAME, missing_roles))
        missing_sites = sorted(
            site.name for site in self.registry
            if int(site.metadata.get("required", 0)) and
            site.name not in self._observations)
        if missing_sites:
            raise RuntimeError("%s required sites were not observed: %s" %
                               (self.MODEL_NAME, missing_sites))
        if self.ALLOWED_CONCAT_CALLS is not None:
            count = sum(
                1 for adapter in self._merge_adapters
                for row in adapter.manifest()
                if row.get("operation") == "concat" or
                "concat" in str(row.get("merge", "")))
            if count not in self.ALLOWED_CONCAT_CALLS:
                raise RuntimeError("%s observed %d concat sites; expected %s" %
                                   (self.MODEL_NAME, count,
                                    self.ALLOWED_CONCAT_CALLS))

    def _sync_merges(self) -> None:
        for adapter in self._merge_adapters:
            for merge in adapter.manifest():
                name = str(merge.get("merge", ""))
                if not name:
                    continue
                operation = str(merge.get("operation", "")) or \
                    ("concat" if "concat" in name else "add")
                site = "merge::" + name
                if site not in self.registry:
                    self._register(
                        site, "concat_merge" if operation == "concat"
                        else "residual_merge", name, False, "runtime_merge",
                        {"operation": operation,
                         "policy": merge.get("policy", self.merge_policy),
                         "operational": 1})
                self._observations.setdefault(site, {
                    "observations": 1, "numel": 0, "nonfinite": 0,
                    "shapes": set(), "dtypes": set(),
                })

    def freeze(self, bits: int) -> None:
        for adapter in self._merge_adapters:
            adapter.freeze(bits)
        self._sync_merges()
        if self.strict and not self._quantization_delegated:
            self._validate()
        if not self.registry.frozen:
            self.registry.freeze()
        self.mode = "bypass"

    def quantize(self) -> None:
        if self._quantization_delegated:
            self.mode = "bypass"
            return
        for adapter in self._merge_adapters:
            adapter.quantize()
        self.mode = "quantize"

    def disable(self) -> None:
        for adapter in self._merge_adapters:
            adapter.disable()
        self.mode = "bypass"
        self._task_capture_enabled = False
        self._task_capture_values = {}
        self._task_signal_values = {}

    def manifest(self) -> List[Dict[str, Any]]:
        return [row for adapter in self._merge_adapters
                for row in adapter.manifest()]

    def semantic_manifest(self) -> List[Dict[str, Any]]:
        active = {"merge::" + str(row.get("merge")): row
                  for row in self.manifest() if row.get("merge")}
        rows = []
        for site in self.registry:
            row = site.manifest()
            for key, value in active.get(site.name, {}).items():
                if key in {"operation", "policy", "bits", "axis",
                           "group_size", "branches", "scale", "scales"}:
                    row["active_" + key] = value
            obs = self._observations.get(site.name)
            row.update({
                "model": self.MODEL_NAME,
                "observed": int(obs is not None),
                "observations": 0 if obs is None else obs["observations"],
                "numel": 0 if obs is None else obs["numel"],
                "nonfinite": 0 if obs is None else obs["nonfinite"],
                "shapes": "" if obs is None else ";".join(sorted(obs["shapes"])),
                "dtypes": "" if obs is None else ";".join(sorted(obs["dtypes"])),
            })
            rows.append(row)
        return rows

    def edge_statistics(self) -> List[Dict[str, Any]]:
        return self.runtime.statistics()

    def close(self) -> None:
        self._task_capture_enabled = False
        self._task_capture_forwards = 0
        self._task_capture_values = {}
        self.disable()
        for adapter in self._merge_adapters:
            adapter.close()
        for handle in self._handles:
            handle.remove()
        self._handles = []
