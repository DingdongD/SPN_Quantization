import unittest

import torch

from spn_quant.runtime import (
    EdgeAwareInstrumentorAdapter,
    EdgeAwareQuantizerProxy,
    EdgeQDQRuntime,
)


class CountingQuantizer(object):
    def __init__(self):
        self.calls = 0
        self.scale = 1.0
        self.bits = 4
        self.qmin = -7
        self.qmax = 7

    def quantize_with_codes(self, tensor):
        self.calls += 1
        quantized = torch.round(tensor)
        return quantized, quantized.to(torch.int32)


class FakeInstrumentor(object):
    def __init__(self):
        self.model = torch.nn.Identity()
        self.activation_mode = "uniform"
        self.quantizers = {}
        self.relu_quantizers = {}

    def configure(self):
        self.quantizers = {
            ("producer", "output"): CountingQuantizer(),
            ("consumer", "input"): CountingQuantizer(),
        }
        self.relu_quantizers = {}

    def metadata(self):
        return {"activation_mode": "uniform"}

    def close(self):
        pass


class EdgeQDQRuntimeTest(unittest.TestCase):
    def test_same_site_and_fanout_reuse_one_quantized_tensor(self):
        runtime = EdgeQDQRuntime()
        quantizer = CountingQuantizer()
        runtime.begin_forward()
        value = torch.tensor([0.6])
        first, _ = runtime.process_with_codes(
            "producer", value, quantizer.quantize_with_codes)
        same, _ = runtime.process_with_codes(
            "producer", value, quantizer.quantize_with_codes)
        consumer, codes = runtime.process_with_codes(
            "consumer", first, quantizer.quantize_with_codes)
        self.assertIs(first, same)
        self.assertIs(first, consumer)
        self.assertIsNone(codes)
        self.assertEqual(quantizer.calls, 1)

    def test_explicit_merge_boundary_forces_requantization(self):
        runtime = EdgeQDQRuntime()
        quantizer = CountingQuantizer()
        runtime.begin_forward()
        value = torch.tensor([0.6])
        first = runtime.process("producer", value, quantizer)
        second = runtime.process("merge", first, quantizer, force=True)
        self.assertEqual(quantizer.calls, 2)
        self.assertEqual(float(second.item()), 1.0)

    def test_proxy_delegates_qparams(self):
        runtime = EdgeQDQRuntime()
        base = CountingQuantizer()
        proxy = EdgeAwareQuantizerProxy(base, runtime, "edge")
        runtime.begin_forward()
        proxy(torch.tensor([0.4]))
        self.assertEqual(proxy.scale, 1.0)
        self.assertEqual(base.calls, 1)

    def test_instrumentor_adapter_wraps_uniform_quantizers(self):
        fake = FakeInstrumentor()
        adapter = EdgeAwareInstrumentorAdapter(fake)
        adapter.configure()
        self.assertIsInstance(
            fake.quantizers[("producer", "output")], EdgeAwareQuantizerProxy)
        metadata = adapter.metadata()
        self.assertEqual(metadata["qdq_semantics"], "logical_edge_once")
        adapter.close()


if __name__ == "__main__":
    unittest.main()
