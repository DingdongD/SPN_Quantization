#!/usr/bin/env python3
"""Shared-scale branch merge support for standard integer backends."""

from __future__ import division

import types

from scripts.hardware_aligned_quantization import HardwareMinMaxObserver


class SharedMergeQuantizer(object):
    def __init__(self, unsigned):
        self.unsigned = bool(unsigned)
        self.observer = HardwareMinMaxObserver()
        self.bits = None
        self.quantizer = None

    @property
    def scale(self):
        if self.quantizer is None:
            raise RuntimeError("merge quantizer is not frozen")
        return self.quantizer.scale

    def observe(self, branches):
        for branch in branches:
            self.observer.update(branch)

    def freeze(self, bits):
        self.bits = int(bits)
        self.quantizer = self.observer.quantizer(bits, unsigned=self.unsigned)

    def quantize(self, branches):
        if self.quantizer is None:
            raise RuntimeError("merge quantizer is not frozen")
        return tuple(self.quantizer(branch) for branch in branches)

    def qparams(self):
        if self.quantizer is None:
            raise RuntimeError("merge quantizer is not frozen")
        return {
            "bits": self.bits,
            "unsigned": self.unsigned,
            "qmin": self.quantizer.qmin,
            "qmax": self.quantizer.qmax,
            "scale": self.quantizer.scale,
            "zero_point": self.quantizer.zero_point,
        }


class CallIndexedConcatAdapter(object):
    """QDQ each executed `_concat` call with a stage-specific shared scale."""

    def __init__(self, model):
        self.model = model
        self.mode = "bypass"
        self.bits = None
        self.call_counts = {}
        self.observers = {}
        self.quantizers = {}
        self.originals = {}
        self.handle = model.register_forward_pre_hook(self._reset_calls)

        for name, module in model.named_modules():
            method = getattr(module, "_concat", None)
            if method is None or not callable(method):
                continue
            self.originals[name] = (module, method)
            module._concat = types.MethodType(self._make_wrapper(name, method), module)

    def _reset_calls(self, module, inputs):
        del module, inputs
        self.call_counts = {}

    def _make_wrapper(self, name, original):
        def wrapper(module, *args, **kwargs):
            del module
            output = original(*args, **kwargs)
            index = self.call_counts.get(name, 0)
            self.call_counts[name] = index + 1
            key = "%s._concat#%d" % (name, index) if name else "_concat#%d" % index
            if self.mode == "observe":
                observer = self.observers.setdefault(key, HardwareMinMaxObserver())
                observer.update(output)
                return output
            if self.mode == "quantize":
                quantizer = self.quantizers.get(key)
                if quantizer is None:
                    raise RuntimeError("unobserved concat call: %s" % key)
                return quantizer.quantize((output,))[0]
            return output
        return wrapper

    def observe(self):
        self.mode = "observe"

    def freeze(self, bits):
        self.bits = int(bits)
        self.quantizers = {}
        for key, observer in self.observers.items():
            merge = SharedMergeQuantizer(unsigned=observer.minimum >= 0.0)
            merge.observer = observer
            merge.freeze(bits)
            self.quantizers[key] = merge
        self.mode = "bypass"

    def quantize(self):
        if not self.originals:
            self.mode = "bypass"
            return
        if not self.quantizers:
            raise RuntimeError("concat calibration must be frozen first")
        self.mode = "quantize"

    def disable(self):
        self.mode = "bypass"

    def manifest(self):
        return [dict({"merge": key, "kind": "concat"}, **quantizer.qparams())
                for key, quantizer in sorted(self.quantizers.items())]

    def close(self):
        self.disable()
        self.handle.remove()
        for name, (module, original) in self.originals.items():
            del name
            module._concat = original
        self.originals = {}
