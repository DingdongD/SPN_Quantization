import unittest

import torch
import torch.nn as nn

from spn_quant import QuantSite, QuantSiteRegistry, QuantSpec
from spn_quant.sites import build_module_site_registry, trace_module_tensor_edges


class FanoutModel(nn.Module):
    def __init__(self):
        super(FanoutModel, self).__init__()
        self.stem = nn.Conv2d(1, 2, 1, bias=False)
        self.left = nn.Conv2d(2, 1, 1, bias=False)
        self.right = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, value):
        shared = self.stem(value)
        return self.left(shared) + self.right(shared)


class QuantSiteRegistryTest(unittest.TestCase):
    def test_registry_fails_closed_on_duplicates_and_unknown_sites(self):
        spec = QuantSpec.signed_tensor(4)
        site = QuantSite("edge", "encoder_activation", "stem#0", ("left#0",), spec)
        registry = QuantSiteRegistry([site])
        with self.assertRaises(ValueError):
            registry.register(site)
        with self.assertRaises(KeyError):
            registry.require("missing")

    def test_registry_cannot_change_after_freeze(self):
        registry = QuantSiteRegistry()
        registry.freeze()
        with self.assertRaises(RuntimeError):
            registry.register(QuantSite(
                "edge", "encoder_activation", "stem#0", (),
                QuantSpec.signed_tensor(4)))

    def test_execution_trace_collapses_fanout_into_one_logical_site(self):
        model = FanoutModel().eval()
        sample = torch.ones(1, 1, 2, 2)
        edges = trace_module_tensor_edges(model, (sample,))
        stem = [edge for edge in edges if edge.producer == "stem#0"][0]
        self.assertEqual(stem.consumers, ("left#0", "right#0"))

    def test_module_registry_assigns_semantic_roles_and_manifest(self):
        model = FanoutModel().eval()
        sample = torch.ones(1, 1, 2, 2)
        group_fn = lambda name, module: "encoder" if name == "stem" else "depth_head"
        registry = build_module_site_registry(model, (sample,), group_fn)
        site = registry.require("activation::stem#0")
        self.assertEqual(site.role, "encoder_activation")
        self.assertEqual(site.consumers, ("left#0", "right#0"))
        self.assertEqual(site.spec.transform, "none")
        self.assertEqual(site.manifest()["fanout"], 2)


if __name__ == "__main__":
    unittest.main()
