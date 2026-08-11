"""Official-aligned learnable activation quantization for QDrop."""

from __future__ import annotations

import hashlib
import json
import math

import torch
import torch.nn as nn


ACTIVATION_CONTRACT_VERSION = 1


def round_ste(value):
    return (torch.round(value) - value).detach() + value


def gradient_scale(value, factor):
    return (value - value * float(factor)).detach() + value * float(factor)


def _contract_fingerprint(entry):
    scale = torch.as_tensor(entry["scale"]).detach().cpu().float().contiguous()
    payload = {
        "format_version": int(entry["format_version"]),
        "site": str(entry["site"]),
        "bits": int(entry["bits"]),
        "signed": int(entry["signed"]),
        "symmetric": int(entry["symmetric"]),
        "qmin": int(entry["qmin"]),
        "qmax": int(entry["qmax"]),
        "zero_point": int(entry["zero_point"]),
        "scale_minimum": float(entry["scale_minimum"]),
        "scale_shape": list(scale.shape),
    }
    header = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + scale.numpy().tobytes()).hexdigest()


def _required_contract_fields():
    return {
        "format_version", "site", "bits", "signed", "symmetric",
        "qmin", "qmax", "scale", "zero_point", "scale_minimum",
        "fingerprint",
    }


class QDropActivationQuantizer(nn.Module):
    """Learn A4 parameters and randomly retain quantized activations."""

    def __init__(self, site, bits, signed, symmetric,
                 scale_minimum, seed):
        super().__init__()
        self.site = str(site)
        self.bits = int(bits)
        self.signed = bool(signed)
        self.unsigned = not self.signed
        self.symmetric = bool(symmetric)
        self.format = "uniform"
        self.scale_minimum = float(scale_minimum)
        self.seed = int(seed)
        if not self.site:
            raise ValueError("QDrop activation site cannot be empty")
        if self.bits != 4:
            raise ValueError("QDrop activation quantizer requires four bits")
        if self.symmetric and not self.signed:
            raise ValueError("unsigned QDrop activations use affine quantization")
        if not math.isfinite(self.scale_minimum) or \
                self.scale_minimum <= 0.0:
            raise ValueError("scale_minimum must be positive and finite")
        if self.symmetric:
            self.qmin = -(2 ** (self.bits - 1) - 1)
            self.qmax = 2 ** (self.bits - 1) - 1
        elif self.signed:
            self.qmin = -(2 ** (self.bits - 1))
            self.qmax = 2 ** (self.bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** self.bits - 1
        self.scale_parameter = nn.Parameter(torch.tensor(1.0))
        if self.symmetric:
            self.register_parameter("zero_point_parameter", None)
        else:
            self.zero_point_parameter = nn.Parameter(torch.tensor(0.0))
        self.phase = "uninitialized"
        self.quant_probability = 0.0
        self._generator = None
        self._generator_device = ""
        self._calls = 0
        self._numel = 0
        self._zero_codes = 0
        self._saturated_codes = 0
        self._squared_error = 0.0
        self._reference_energy = 0.0

    @property
    def scale(self):
        return float(self._hard_scale().detach().cpu().item())

    @property
    def zero_point(self):
        return int(self._hard_zero_point().detach().cpu().item())

    def scale_for(self, tensor):
        return self._scale(tensor) \
            if self.phase == "reconstruction" else self._hard_scale()

    def initialize(self, tensor):
        if self.phase != "uninitialized":
            raise RuntimeError("QDrop activation quantizer is already initialized")
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("QDrop activation calibration requires a tensor")
        detached = tensor.detach().float()
        if not bool(torch.isfinite(detached).all().item()):
            raise ValueError("QDrop activation calibration must be finite")
        minimum = float(detached.min().item())
        maximum = float(detached.max().item())
        if self.symmetric:
            scale = max(abs(minimum), abs(maximum)) / float(self.qmax)
            zero_point = 0.0
        else:
            scale = (maximum - minimum) / float(self.qmax - self.qmin)
            scale = max(scale, self.scale_minimum)
            zero_point = round(self.qmin - minimum / scale)
            zero_point = min(max(zero_point, self.qmin), self.qmax)
        scale = max(scale, self.scale_minimum)
        with torch.no_grad():
            self.scale_parameter.copy_(
                torch.tensor(
                    scale,
                    device=self.scale_parameter.device,
                    dtype=self.scale_parameter.dtype))
            if self.zero_point_parameter is not None:
                self.zero_point_parameter.copy_(
                    torch.tensor(
                        zero_point,
                        device=self.zero_point_parameter.device,
                        dtype=self.zero_point_parameter.dtype))
        self.phase = "initialized"

    def start_reconstruction(self, quant_probability):
        if self.phase != "initialized":
            raise RuntimeError(
                "QDrop reconstruction requires initialized activation parameters")
        self.set_quant_probability(quant_probability)
        self.scale_parameter.requires_grad_(True)
        if self.zero_point_parameter is not None:
            self.zero_point_parameter.requires_grad_(True)
        self._generator = None
        self._generator_device = ""
        self.phase = "reconstruction"

    def set_quant_probability(self, quant_probability):
        if self.phase not in ("initialized", "reconstruction"):
            raise RuntimeError(
                "QDrop probability requires initialized reconstruction state")
        probability = float(quant_probability)
        if not math.isfinite(probability) or \
                probability < 0.0 or probability > 1.0:
            raise ValueError("quant_probability must be in [0, 1]")
        self.quant_probability = probability

    def _scale(self, tensor):
        factor = 1.0 / math.sqrt(
            float(tensor.numel() * max(abs(self.qmin), abs(self.qmax))))
        value = self.scale_parameter.abs().clamp_min(self.scale_minimum)
        return gradient_scale(value, factor)

    def _zero_point(self, tensor):
        if self.zero_point_parameter is None:
            return tensor.new_tensor(0.0)
        factor = 1.0 / math.sqrt(float(tensor.numel() * (self.qmax - self.qmin)))
        value = gradient_scale(self.zero_point_parameter, factor)
        return round_ste(value).clamp(self.qmin, self.qmax)

    def _hard_scale(self):
        return self.scale_parameter.abs().clamp_min(self.scale_minimum)

    def _hard_zero_point(self):
        if self.zero_point_parameter is None:
            return self.scale_parameter.new_tensor(0.0)
        return torch.round(self.zero_point_parameter).clamp(
            self.qmin, self.qmax)

    def _deterministic(self, tensor):
        scale = self._scale(tensor) \
            if self.phase == "reconstruction" else self._hard_scale()
        zero_point = self._zero_point(tensor) \
            if self.phase == "reconstruction" else self._hard_zero_point()
        transformed = tensor / scale + zero_point
        soft_codes = round_ste(transformed).clamp(self.qmin, self.qmax)
        quantized = (soft_codes - zero_point) * scale
        hard_codes = torch.round(transformed.detach()).clamp(
            self.qmin, self.qmax)
        code_dtype = torch.int8 if self.signed else torch.uint8
        return quantized, hard_codes.to(code_dtype)

    def _mask(self, tensor):
        device = tensor.device.type
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
            self._generator_device = device
        if self._generator_device != device:
            raise RuntimeError("QDrop activation generator device changed")
        return torch.rand(
            tensor.shape,
            dtype=torch.float32,
            device=tensor.device,
            generator=self._generator) < self.quant_probability

    def quantize_with_codes(self, tensor):
        if self.phase not in ("reconstruction", "frozen"):
            raise RuntimeError("QDrop activation quantizer is not active")
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("QDrop activation quantizer requires a floating tensor")
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError("QDrop activation tensor must be finite")
        deterministic, codes = self._deterministic(tensor)
        output = deterministic
        if self.phase == "reconstruction":
            output = torch.where(self._mask(tensor), deterministic, tensor)
        detached_codes = codes.detach()
        error = deterministic.detach().float() - tensor.detach().float()
        self._calls += 1
        self._numel += int(codes.numel())
        self._zero_codes += int((detached_codes == self.zero_point).sum().item())
        self._saturated_codes += int(
            ((detached_codes == self.qmin) |
             (detached_codes == self.qmax)).sum().item())
        self._squared_error += float(error.square().sum().item())
        self._reference_energy += float(
            tensor.detach().float().square().sum().item())
        return output, codes

    def freeze(self):
        if self.phase != "reconstruction":
            raise RuntimeError("QDrop freeze requires reconstruction phase")
        if not bool(torch.isfinite(self._hard_scale()).all().item()):
            raise FloatingPointError("QDrop activation scale is non-finite")
        self.scale_parameter.requires_grad_(False)
        if self.zero_point_parameter is not None:
            self.zero_point_parameter.requires_grad_(False)
        self.quant_probability = 1.0
        self._generator = None
        self._generator_device = ""
        self.phase = "frozen"

    def contract(self):
        if self.phase != "frozen":
            raise RuntimeError("QDrop activation contract requires frozen state")
        entry = {
            "format_version": ACTIVATION_CONTRACT_VERSION,
            "site": self.site,
            "bits": self.bits,
            "signed": int(self.signed),
            "symmetric": int(self.symmetric),
            "qmin": self.qmin,
            "qmax": self.qmax,
            "scale": self._hard_scale().detach().cpu().float().clone(),
            "zero_point": self.zero_point,
            "scale_minimum": self.scale_minimum,
        }
        entry["fingerprint"] = _contract_fingerprint(entry)
        return entry

    def statistics(self):
        zero_ratio = float(self._zero_codes) / max(self._numel, 1)
        saturation_ratio = float(self._saturated_codes) / max(self._numel, 1)
        sqnr = float("inf")
        if self._squared_error > 0.0:
            sqnr = 10.0 * math.log10(
                max(self._reference_energy, self.scale_minimum) /
                self._squared_error)
        return {
            "site": self.site,
            "calls": self._calls,
            "numel": self._numel,
            "zero_ratio": zero_ratio,
            "saturation_ratio": saturation_ratio,
            "sqnr_db": sqnr,
        }


class ExactActivationQuantizer(object):
    """Deterministic activation QDQ reconstructed from an exact contract."""

    def __init__(self, entry):
        self.entry = dict(entry)
        self.site = str(self.entry["site"])
        self.bits = int(self.entry["bits"])
        self.signed = bool(int(self.entry["signed"]))
        self.unsigned = not self.signed
        self.symmetric = bool(int(self.entry["symmetric"]))
        self.format = "uniform"
        self.qmin = int(self.entry["qmin"])
        self.qmax = int(self.entry["qmax"])
        self.scale_tensor = torch.as_tensor(
            self.entry["scale"]).detach().cpu().float().clone()
        self.zero_point = int(self.entry["zero_point"])
        self.phase = "frozen"

    @property
    def scale(self):
        return float(self.scale_tensor.item())

    def scale_for(self, tensor):
        return self.scale_tensor.to(device=tensor.device, dtype=tensor.dtype)

    @classmethod
    def from_contract(cls, entry):
        fields = set(entry)
        required = _required_contract_fields()
        if fields != required:
            raise KeyError(
                "QDrop activation contract fields mismatch: missing=%s extra=%s" %
                (sorted(required - fields), sorted(fields - required)))
        if int(entry["format_version"]) != ACTIVATION_CONTRACT_VERSION:
            raise ValueError("unsupported QDrop activation contract version")
        if int(entry["bits"]) != 4:
            raise ValueError("QDrop activation contract requires four bits")
        if _contract_fingerprint(entry) != str(entry["fingerprint"]):
            raise RuntimeError("QDrop activation contract fingerprint mismatch")
        scale = torch.as_tensor(entry["scale"]).float()
        if scale.numel() != 1 or \
                not bool(torch.isfinite(scale).all().item()) or \
                float(scale.item()) <= 0.0:
            raise ValueError("QDrop activation contract scale is invalid")
        if int(entry["zero_point"]) < int(entry["qmin"]) or \
                int(entry["zero_point"]) > int(entry["qmax"]):
            raise ValueError("QDrop activation zero point is outside its range")
        return cls(entry)

    def quantize_with_codes(self, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError("exact activation QDQ requires a floating tensor")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("exact activation tensor must be finite")
        scale = self.scale_tensor.to(device=tensor.device, dtype=tensor.dtype)
        transformed = tensor / scale + float(self.zero_point)
        code_dtype = torch.int8 if self.signed else torch.uint8
        codes = torch.round(transformed).clamp(
            self.qmin, self.qmax).to(code_dtype)
        quantized = (
            codes.to(tensor.dtype) - float(self.zero_point)) * scale
        return quantized, codes

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]
