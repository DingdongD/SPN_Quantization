"""Training policies, objective, and metrics for NLSPN scene fine-tuning."""

import math

import numpy as np
import torch


ENCODER_PREFIXES = (
    "conv1_rgb.", "conv2.", "conv3.", "conv4.", "conv5.", "conv6.")
ADAPTATION_PREFIXES = (
    "conv1_dep.", "dec5.", "dec4.", "dec3.", "dec2.",
    "id_dec1.", "id_dec0.", "gd_dec1.", "gd_dec0.",
    "cf_dec1.", "cf_dec0.", "prop_layer.")

DEPTH_BANDS = (
    ("band_0_2", 0.0, 2.0),
    ("band_2_4", 2.0, 4.0),
    ("band_4_6", 4.0, 6.0),
    ("band_6_8", 6.0, 8.0),
    ("band_8_10", 8.0, 10.0),
)
DEPTH_BAND_NAMES = tuple(item[0] for item in DEPTH_BANDS)


def trainable_parameter_names(model):
    return tuple(name for name, parameter in model.named_parameters()
                 if parameter.requires_grad)


def original_trainable_names(model):
    attribute = "_nlspn_original_trainable_names"
    if not hasattr(model, attribute):
        setattr(model, attribute, tuple(trainable_parameter_names(model)))
    return tuple(getattr(model, attribute))


def _batch_norm_parameter_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        for parameter_name, _ in module.named_parameters(recurse=False):
            prefix = module_name + "." if module_name else ""
            names.add(prefix + parameter_name)
    return names


def keep_batch_norm_frozen(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = False


def freeze_batch_norm(model):
    keep_batch_norm_frozen(model)


def _matches_any(name, prefixes):
    return any(name.startswith(prefix) for prefix in prefixes)


def assert_trainable_set(model, groups):
    declared = set()
    named = dict(model.named_parameters())
    for group in groups:
        names = tuple(group.get("parameter_names", ()))
        parameters = tuple(group.get("params", ()))
        if len(names) != len(parameters):
            raise ValueError("declared and actual trainable parameter mismatch")
        for name, parameter in zip(names, parameters):
            if name not in named or named[name] is not parameter:
                raise ValueError("declared and actual trainable parameter mismatch")
        if declared.intersection(names):
            raise ValueError("declared trainable groups overlap")
        declared.update(names)
    actual = set(trainable_parameter_names(model))
    if declared != actual:
        raise ValueError(
            "declared and actual trainable parameter mismatch: {} != {}".format(
                sorted(declared), sorted(actual)))


def build_exact_groups(model, eligible, spec):
    named = dict(model.named_parameters())
    missing = sorted(set(eligible) - set(named))
    if missing:
        raise ValueError("original trainable parameters disappeared: {}".format(missing))

    batch_norm = _batch_norm_parameter_names(model)
    eligible_non_bn = tuple(name for name in eligible if name not in batch_norm)
    all_prefixes = ENCODER_PREFIXES + ADAPTATION_PREFIXES
    unknown = sorted(name for name in eligible_non_bn
                     if not _matches_any(name, all_prefixes))
    if unknown:
        raise ValueError("unknown trainable parameter prefixes: {}".format(unknown))

    for parameter in model.parameters():
        parameter.requires_grad = False

    groups = []
    selected = set()
    for group_name, prefixes, learning_rate in spec:
        names = tuple(name for name in eligible_non_bn
                      if _matches_any(name, prefixes))
        if not names:
            raise ValueError("{} parameter group is empty".format(group_name))
        overlap = selected.intersection(names)
        if overlap:
            raise ValueError("trainable parameter groups overlap: {}".format(overlap))
        parameters = [named[name] for name in names]
        for parameter in parameters:
            parameter.requires_grad = True
        selected.update(names)
        groups.append({
            "name": group_name,
            "params": parameters,
            "parameter_names": names,
            "lr": float(learning_rate),
        })

    freeze_batch_norm(model)
    assert_trainable_set(model, groups)
    return groups


def configure_stage(model, stage):
    if stage == 1:
        spec = (("adaptation", ADAPTATION_PREFIXES, 1e-4),)
    elif stage == 2:
        spec = (
            ("encoder", ENCODER_PREFIXES, 5e-6),
            ("adaptation", ADAPTATION_PREFIXES, 2e-5),
        )
    else:
        raise ValueError("stage must be 1 or 2")
    eligible = original_trainable_names(model)
    return build_exact_groups(model, eligible, spec)


def _validated_errors(prediction, target, max_depth):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes do not match")
    if max_depth <= 0:
        raise ValueError("max depth must be positive")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction or target contains nonfinite values")
    valid = (target > 0.0) & (target <= float(max_depth))
    if not torch.any(valid):
        raise ValueError("target contains no valid pixels")
    target_valid = target[valid]
    prediction_valid = prediction.clamp(0.0, float(max_depth))[valid]
    return prediction_valid - target_valid, target_valid, valid


def masked_l1_l2(prediction, target, max_depth=10.0):
    error, _, _ = _validated_errors(prediction, target, max_depth)
    return error.abs().mean() + error.square().mean()


def frame_error_sums(prediction, target, max_depth=10.0):
    error, target_valid, valid = _validated_errors(
        prediction, target, max_depth)
    absolute = error.abs().to(torch.float64)
    squared = error.to(torch.float64).square()
    target_double = target_valid.to(torch.float64)
    result = {
        "squared_error_sum": np.float64(squared.sum().item()),
        "absolute_error_sum": np.float64(absolute.sum().item()),
        "abs_rel_sum": np.float64((absolute / target_double).sum().item()),
        "valid_pixel_count": int(valid.sum().item()),
    }

    flat_target = target[valid]
    flat_error = error
    for name, lower, upper in DEPTH_BANDS:
        band = (flat_target > lower) & (flat_target <= upper)
        band_error = flat_error[band].to(torch.float64)
        result[name + "_squared_error_sum"] = np.float64(
            band_error.square().sum().item())
        result[name + "_absolute_error_sum"] = np.float64(
            band_error.abs().sum().item())
        result[name + "_valid_pixel_count"] = int(band.sum().item())

    band_count = sum(result[name + "_valid_pixel_count"]
                     for name in DEPTH_BAND_NAMES)
    if band_count != result["valid_pixel_count"]:
        raise ValueError("depth-band counts do not cover all valid pixels")
    return result


def _new_raw_sums():
    return {
        "squared_error_sum": np.float64(0.0),
        "absolute_error_sum": np.float64(0.0),
        "abs_rel_sum": np.float64(0.0),
        "valid_pixel_count": 0,
    }


def _add_raw_sums(destination, squared, absolute, abs_rel, count):
    values = np.asarray([squared, absolute, abs_rel], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("metric update contains nonfinite values")
    if np.any(values < 0.0):
        raise ValueError("metric raw sums must be nonnegative")
    count = int(count)
    if count <= 0:
        raise ValueError("metric valid-pixel count must be positive")
    destination["squared_error_sum"] += values[0]
    destination["absolute_error_sum"] += values[1]
    destination["abs_rel_sum"] += values[2]
    destination["valid_pixel_count"] += count


def _summarize(raw):
    count = raw["valid_pixel_count"]
    if count <= 0:
        raise ValueError("cannot summarize metrics with no valid pixels")
    return {
        "squared_error_sum": np.float64(raw["squared_error_sum"]),
        "absolute_error_sum": np.float64(raw["absolute_error_sum"]),
        "abs_rel_sum": np.float64(raw["abs_rel_sum"]),
        "valid_pixel_count": int(count),
        "rmse": math.sqrt(float(raw["squared_error_sum"]) / count),
        "mae": float(raw["absolute_error_sum"]) / count,
        "abs_rel": float(raw["abs_rel_sum"]) / count,
    }


class MetricAccumulator(object):
    def __init__(self):
        self._pooled = _new_raw_sums()
        self._scenes = {}
        self._bands = {name: _new_raw_sums() for name in DEPTH_BAND_NAMES}

    def add(self, scene, squared_error_sum, absolute_error_sum,
            abs_rel_sum, valid_pixel_count, bands=None):
        scene = str(scene)
        if not scene:
            raise ValueError("metric scene must be nonempty")
        if scene not in self._scenes:
            self._scenes[scene] = _new_raw_sums()
        _add_raw_sums(
            self._pooled, squared_error_sum, absolute_error_sum,
            abs_rel_sum, valid_pixel_count)
        _add_raw_sums(
            self._scenes[scene], squared_error_sum, absolute_error_sum,
            abs_rel_sum, valid_pixel_count)
        if bands is not None:
            for name in DEPTH_BAND_NAMES:
                values = bands[name]
                count = int(values["valid_pixel_count"])
                if count == 0:
                    continue
                _add_raw_sums(
                    self._bands[name], values["squared_error_sum"],
                    values["absolute_error_sum"], values["abs_rel_sum"], count)

    def add_frame(self, scene, prediction, target, max_depth=10.0):
        raw = frame_error_sums(prediction, target, max_depth)
        valid_target = target[(target > 0.0) & (target <= max_depth)]
        clamped = prediction.clamp(0.0, max_depth)[
            (target > 0.0) & (target <= max_depth)]
        absolute = (clamped - valid_target).abs().to(torch.float64)
        bands = {}
        for name, lower, upper in DEPTH_BANDS:
            mask = (valid_target > lower) & (valid_target <= upper)
            bands[name] = {
                "squared_error_sum": raw[name + "_squared_error_sum"],
                "absolute_error_sum": raw[name + "_absolute_error_sum"],
                "abs_rel_sum": np.float64(
                    (absolute[mask] / valid_target[mask].to(torch.float64)).sum().item()),
                "valid_pixel_count": raw[name + "_valid_pixel_count"],
            }
        self.add(
            scene, raw["squared_error_sum"], raw["absolute_error_sum"],
            raw["abs_rel_sum"], raw["valid_pixel_count"], bands=bands)
        return raw

    def finalize(self):
        if self._pooled["valid_pixel_count"] <= 0:
            raise ValueError("no metrics have been accumulated")
        pooled = _summarize(self._pooled)
        scenes = {name: _summarize(raw)
                  for name, raw in sorted(self._scenes.items())}
        result = dict(pooled)
        result.update({
            "pooled_rmse": pooled["rmse"],
            "pooled_mae": pooled["mae"],
            "pooled_abs_rel": pooled["abs_rel"],
            "scene_macro_rmse": float(np.mean(
                [item["rmse"] for item in scenes.values()])),
            "scene_macro_mae": float(np.mean(
                [item["mae"] for item in scenes.values()])),
            "scene_macro_abs_rel": float(np.mean(
                [item["abs_rel"] for item in scenes.values()])),
            "scenes": scenes,
        })
        for name in DEPTH_BAND_NAMES:
            raw = self._bands[name]
            result[name] = None if raw["valid_pixel_count"] == 0 else _summarize(raw)
        return result
