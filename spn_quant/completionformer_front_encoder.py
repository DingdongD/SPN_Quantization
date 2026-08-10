"""CompletionFormer front-encoder W8A8 allocation contract."""

import torch
from torch import nn

FRONT_ENCODER_UNIT_ORDER = (
    "Stem",
    "Embed1.0",
    "Embed1.1",
    "Embed1.2",
    "Embed2.0",
    "Embed2.1",
    "Embed2.2",
    "Embed2.3",
    "PatchEmbed1",
)


def _block_contract(stage, index, downsample):
    prefix = "backbone.former.embed_layer%d.%d" % (stage, index)
    weights = (
        prefix + ".conv1",
        prefix + ".conv2",
        prefix + ".ca.fc.0",
        prefix + ".ca.fc.2",
        prefix + ".sa.conv1",
    )
    if downsample:
        weights += (prefix + ".downsample.0",)
    activations = weights + (
        prefix + ".relu#0",
        prefix + ".relu#1",
        prefix + ".ca.fc.1#0",
        prefix + ".ca.fc.1#1",
    )
    return {
        "weight_modules": weights,
        "activation_sites": activations,
    }


def _expected_units():
    return {
        "Stem": {
            "weight_modules": (
                "backbone.conv1_rgb.0",
                "backbone.conv1_dep.0",
                "backbone.conv1.0",
            ),
            "activation_sites": (
                "backbone.conv1_rgb.0",
                "backbone.conv1_dep.0",
                "backbone.conv1.0",
                "backbone.conv1_rgb.1#0",
                "backbone.conv1_dep.1#0",
                "backbone.conv1.1#0",
            ),
        },
        "Embed1.0": _block_contract(1, 0, False),
        "Embed1.1": _block_contract(1, 1, False),
        "Embed1.2": _block_contract(1, 2, False),
        "Embed2.0": _block_contract(2, 0, True),
        "Embed2.1": _block_contract(2, 1, False),
        "Embed2.2": _block_contract(2, 2, False),
        "Embed2.3": _block_contract(2, 3, False),
        "PatchEmbed1": {
            "weight_modules": (
                "backbone.former.patch_embed1.proj",
            ),
            "activation_sites": (
                "backbone.former.patch_embed1.proj",
                "backbone.former.patch_embed1.norm",
            ),
        },
    }


def _is_front_weight(name):
    return name in (
        "backbone.conv1_rgb.0",
        "backbone.conv1_dep.0",
        "backbone.conv1.0",
    ) or name.startswith(
        "backbone.former.embed_layer1.") or name.startswith(
        "backbone.former.embed_layer2.") or name.startswith(
        "backbone.former.patch_embed1.")


def _is_front_activation(name):
    return name.startswith("backbone.conv1_rgb.") or         name.startswith("backbone.conv1_dep.") or         name.startswith("backbone.conv1.") or         name.startswith("backbone.former.embed_layer1.") or         name.startswith("backbone.former.embed_layer2.") or         name.startswith("backbone.former.patch_embed1.")


def resolve_front_encoder_units(weight_modules, activation_sites):
    expected = _expected_units()
    expected_weights = set()
    expected_activations = set()
    for unit in FRONT_ENCODER_UNIT_ORDER:
        expected_weights.update(expected[unit]["weight_modules"])
        expected_activations.update(expected[unit]["activation_sites"])

    actual_weights = set(
        name for name in weight_modules if _is_front_weight(name))
    actual_activations = set(
        name for name in activation_sites if _is_front_activation(name))
    if actual_weights != expected_weights:
        raise ValueError(
            "front encoder weight modules mismatch: missing=%s extra=%s" % (
                sorted(expected_weights - actual_weights),
                sorted(actual_weights - expected_weights)))
    if actual_activations != expected_activations:
        raise ValueError(
            "front encoder activation sites mismatch: missing=%s extra=%s" % (
                sorted(expected_activations - actual_activations),
                sorted(actual_activations - expected_activations)))

    owners = {}
    units = {}
    for unit in FRONT_ENCODER_UNIT_ORDER:
        weights = expected[unit]["weight_modules"]
        for name in weights:
            if name in owners:
                raise ValueError(
                    "front encoder weight module has multiple owners: %s" %
                    name)
            owners[name] = unit
        units[unit] = {
            "weight_modules": tuple(weights),
            "activation_sites": tuple(expected[unit]["activation_sites"]),
        }
    return units


def _validate_selected_units(units, selected_units):
    selected = tuple(selected_units)
    unknown = set(selected) - set(units)
    if unknown:
        raise ValueError("unknown front encoder units: %s" % sorted(unknown))
    ordered = tuple(
        unit for unit in FRONT_ENCODER_UNIT_ORDER if unit in set(selected))
    if selected != ordered:
        raise ValueError(
            "selected units must follow official unit order: %s" %
            (selected,))
    return selected


def promotion_overrides(units, selected_units):
    selected = _validate_selected_units(units, selected_units)
    weight_overrides = {}
    activation_overrides = {}
    for unit in selected:
        for name in units[unit]["weight_modules"]:
            weight_overrides[name] = 8
        for name in units[unit]["activation_sites"]:
            activation_overrides[name] = 8
    return {
        "weight_bit_overrides": weight_overrides,
        "activation_bit_overrides": activation_overrides,
    }


def unit_manifest_rows(units):
    rows = []
    for unit in FRONT_ENCODER_UNIT_ORDER:
        rows.append({
            "unit": unit,
            "kind": "unit",
            "site": "",
            "bits": "",
        })
        for name in units[unit]["weight_modules"]:
            rows.append({
                "unit": unit,
                "kind": "weight",
                "site": name,
                "bits": 8,
            })
        for name in units[unit]["activation_sites"]:
            rows.append({
                "unit": unit,
                "kind": "activation",
                "site": name,
                "bits": 8,
            })
    return rows


def _module_macs(module, output):
    if not torch.is_tensor(output):
        raise TypeError("quantized module output must be a tensor")
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if output.ndim != 4:
            raise ValueError("convolution output must be NCHW")
        batch = int(output.shape[0])
        spatial = int(output.shape[2] * output.shape[3])
        channels = int(output.shape[1])
        kernel = int(module.kernel_size[0] * module.kernel_size[1])
        inputs_per_group = int(module.in_channels // module.groups)
        return batch * spatial * channels * inputs_per_group * kernel
    if isinstance(module, nn.Linear):
        vectors = int(output.numel() // module.out_features)
        return vectors * int(module.in_features) * int(module.out_features)
    raise TypeError(
        "unsupported quantized cost module: %s" %
        type(module).__name__)


def profile_quantized_costs(model, model_args, quantized_modules,
                            module_to_unit):
    unknown_units = set(module_to_unit) - set(quantized_modules)
    if unknown_units:
        raise ValueError(
            "cost unit mapping has unknown modules: %s" %
            sorted(unknown_units))
    counters = dict(
        (name, {"invocations": 0, "macs": 0})
        for name in quantized_modules)
    handles = []

    def cost_hook(name):
        def hook(module, inputs, output):
            del inputs
            counters[name]["invocations"] += 1
            counters[name]["macs"] += _module_macs(module, output)
        return hook

    for name, module in quantized_modules.items():
        if not isinstance(module, (
                nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            raise TypeError(
                "unsupported quantized cost module: %s" %
                type(module).__name__)
        handles.append(module.register_forward_hook(cost_hook(name)))
    try:
        with torch.no_grad():
            model(*model_args)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for name in sorted(quantized_modules):
        module = quantized_modules[name]
        unit = module_to_unit[name] if name in module_to_unit else ""
        rows.append({
            "module": name,
            "unit": unit,
            "invocations": counters[name]["invocations"],
            "macs": counters[name]["macs"],
            "parameters": int(module.weight.numel()),
            "operators": 1,
        })
    return rows


def _empty_cost():
    return {
        "macs": 0,
        "parameters": 0,
        "operators": 0,
    }


def _add_cost(target, row):
    for key in ("macs", "parameters", "operators"):
        target[key] += int(row[key])


def aggregate_unit_costs(cost_rows, unit_order):
    unit_order = tuple(unit_order)
    if len(unit_order) != len(set(unit_order)):
        raise ValueError("cost unit order contains duplicates")
    units = dict((unit, _empty_cost()) for unit in unit_order)
    whole_model = _empty_cost()
    front_encoder = _empty_cost()
    for row in cost_rows:
        _add_cost(whole_model, row)
        unit = row["unit"]
        if not unit:
            continue
        if unit not in units:
            raise ValueError("unknown cost unit: %s" % unit)
        _add_cost(units[unit], row)
        _add_cost(front_encoder, row)
    return {
        "unit_order": unit_order,
        "units": units,
        "whole_model": whole_model,
        "front_encoder": front_encoder,
    }


def _share(numerator, denominator, label):
    if int(denominator) <= 0:
        raise ValueError("cost denominator must be positive: %s" % label)
    return float(numerator) / float(denominator)


def configuration_cost(costs, selected_units):
    selected = tuple(selected_units)
    unknown = set(selected) - set(costs["unit_order"])
    if unknown:
        raise ValueError("unknown cost units: %s" % sorted(unknown))
    ordered = tuple(
        unit for unit in costs["unit_order"] if unit in set(selected))
    if selected != ordered:
        raise ValueError("selected cost units must follow unit order")
    total = _empty_cost()
    for unit in selected:
        _add_cost(total, costs["units"][unit])
    output = dict(total)
    for metric in ("macs", "parameters", "operators"):
        output["whole_model_%s_share" % (
            "mac" if metric == "macs" else
            "parameter" if metric == "parameters" else "operator")] = \
            _share(total[metric], costs["whole_model"][metric], metric)
        output["front_encoder_%s_share" % (
            "mac" if metric == "macs" else
            "parameter" if metric == "parameters" else "operator")] = \
            _share(total[metric], costs["front_encoder"][metric], metric)
    return output
