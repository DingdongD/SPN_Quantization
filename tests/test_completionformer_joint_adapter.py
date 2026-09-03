import unittest

import torch
import torch.nn as nn

from spn_quant.adapters.completionformer_joint import (
    CompletionFormerJointAdapter,
)
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)
from spn_quant.propagation import PropagationQuantConfig
from spn_quant.qdrop_activation import QDropActivationQuantizer
from spn_quant.qdrop_targets import QDropActivationSite, QDropTargetPlan
from spn_quant.qat.model_methods import (
    ModelHardDeploymentController,
    ModelMethodQATConfig,
)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_heads = 2
        self.scale = 0.5
        self.sr_ratio = 1
        self.q = nn.Linear(8, 8)
        self.kv = nn.Linear(8, 16)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(8, 8)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x, height, width):
        del height, width
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(
            batch, tokens, self.num_heads,
            channels // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(
            batch, tokens, 2, self.num_heads,
            channels // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        probability = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(self.attn_drop(probability), v)
        output = context.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attn = Attention()
        self.concat_conv = nn.Conv2d(16, 8, kernel_size=1, bias=True)

    def forward(self, tokens, cnn, height, width):
        tokens = self.attn(tokens, height, width)
        transformer = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], height, width)
        return self.concat_conv(torch.cat((transformer, cnn), dim=1))


class Former(nn.Module):
    def __init__(self):
        super(Former, self).__init__()
        self.block1 = nn.ModuleList([Block()])

    def forward(self, tokens, cnn, height, width):
        return self.block1[0](tokens, cnn, height, width)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.former = Former()

    def forward(self, tokens, cnn, height, width):
        return self.former(tokens, cnn, height, width)


class ToyCompletionFormer(nn.Module):
    def __init__(self):
        super(ToyCompletionFormer, self).__init__()
        self.backbone = Backbone()

    def forward(self, tokens, cnn, height, width):
        return self.backbone(tokens, cnn, height, width)


def make_inputs():
    torch.manual_seed(19)
    return torch.randn(1, 4, 8), torch.randn(1, 8, 2, 2), 2, 2


def make_adapter(model, expected_attention_modules=1,
                 expected_concat_modules=1):
    return CompletionFormerJointAdapter(
        model=model,
        expected_attention_modules=expected_attention_modules,
        expected_concat_modules=expected_concat_modules,
        weight_bits=4,
        qkv_bits=4,
        probability_bits=8,
        concat_bits=4,
        output_bits=4,
        clip_factors=(1.0,),
        search_rounds=1,
        cache_sample_limit=2,
        cache_byte_limit=1 << 22)


def make_hard_controller(model, adapter):
    owner = "backbone.former.block1.0"
    attention = owner + ".attn"
    concat = owner + ".concat_conv"
    sites = tuple([
        QDropActivationSite(
            site="attention::%s::%s" % (attention, role),
            owner_name=owner,
            owner_kind="attention_qkv",
            role="attention_%s" % role,
            signed=True,
            symmetric=True)
        for role in ("q", "k", "v")
    ] + [
        QDropActivationSite(
            site="concat::%s::%s_input" % (concat, role),
            owner_name=owner,
            owner_kind="concat_input",
            role="concat_%s_input" % role,
            signed=True,
            symmetric=True)
        for role in ("transformer", "cnn")
    ])
    owners = tuple((site.site, site.role) for site in sites)
    contract = QuantizationModelContract(
        model_name="completionformer",
        blocks=(QuantizationBlock(
            owner, (attention + ".q",), owners),),
        prefix_groups=((owner,),),
        tail_groups=((owner,),),
        protected_roles=("affinity",),
        attention_edges=tuple(
            site.site for site in sites
            if site.owner_kind == "attention_qkv"),
        concat_edges=tuple(
            site.site for site in sites
            if site.owner_kind == "concat_input"),
        protected_modules=(),
        module_roles=(),
    )
    plan = QDropTargetPlan(
        model="completionformer",
        blocks=(owner,),
        activation_sites=sites,
        excluded_sites=(),
    )
    config = ModelMethodQATConfig(
        method="lsqplus",
        weight_bits=((attention + ".q", 4),),
        activation_bits=tuple((entry, 4) for entry in owners),
        propagation=PropagationQuantConfig(
            affinity_bits=8,
            confidence_bits=8,
            offset_bits=8,
            state_bits=8,
            coefficient_fraction_bits=13,
        ),
        hawq_range_momentum=0.9,
    )
    qparams = {
        "activation": tuple({
            "owner": entry,
            "bits": 4,
            "unsigned": 0,
            "qmin": -8,
            "qmax": 7,
            "scale": 0.25,
            "offset": 0.0,
        } for entry in owners),
        "propagation": None,
    }
    return ModelHardDeploymentController(
        model, contract, plan, config, qparams, joint_adapter=adapter)


class CompletionFormerJointDiscoveryTest(unittest.TestCase):
    def test_discovers_owned_boundaries_and_restores_forwards(self):
        torch.manual_seed(11)
        model = ToyCompletionFormer().eval()
        attention = model.backbone.former.block1[0].attn
        concat = model.backbone.former.block1[0].concat_conv
        original_attention = attention.forward
        original_concat = concat.forward
        reference = model(*make_inputs())

        adapter = make_adapter(model)
        candidate = model(*make_inputs())

        torch.testing.assert_close(candidate, reference)
        self.assertEqual(adapter.attention_names(), [
            "backbone.former.block1.0.attn"])
        self.assertEqual(adapter.concat_names(), [
            "backbone.former.block1.0.concat_conv"])
        self.assertEqual(adapter.externally_owned_inputs(), [
            "backbone.former.block1.0.concat_conv"])
        self.assertEqual(adapter.externally_owned_outputs(), [
            "backbone.former.block1.0.attn.kv",
            "backbone.former.block1.0.attn.q",
            "backbone.former.block1.0.concat_conv",
        ])

        adapter.close()

        self.assertEqual(attention.forward, original_attention)
        self.assertEqual(concat.forward, original_concat)
        torch.testing.assert_close(model(*make_inputs()), reference)

    def test_rejects_inexact_official_module_counts(self):
        model = ToyCompletionFormer().eval()

        with self.assertRaisesRegex(RuntimeError, "expected 2 Attention"):
            make_adapter(model, expected_attention_modules=2)

    def test_requires_eval_mode(self):
        model = ToyCompletionFormer().train()

        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            make_adapter(model)


class CompletionFormerJointCalibrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = ToyCompletionFormer().eval()
        self.adapter = make_adapter(self.model)
        self.addCleanup(self.adapter.close)
        self.inputs = make_inputs()

    def test_paired_pass_freezes_and_quantizes_both_boundaries(self):
        self.adapter.capture_targets()
        target_output = self.model(*self.inputs)
        self.adapter.observe_reconstruction()
        reconstruction_output = self.model(*self.inputs)

        torch.testing.assert_close(reconstruction_output, target_output)
        self.adapter.freeze()
        calibration = self.adapter.calibration_metadata()
        self.assertEqual(calibration["target_forwards"], 1)
        self.assertEqual(calibration["reconstruction_forwards"], 1)
        self.assertEqual(calibration["attention_modules"], 1)
        self.assertEqual(calibration["concat_modules"], 1)
        self.adapter.configure(
            attention_enabled=True,
            concat_enabled=True,
            qkv_bits=4,
            concat_bits=4,
            output_bits=4)
        quantized = self.model(*self.inputs)

        self.assertTrue(bool(torch.isfinite(quantized).all().item()))
        self.assertEqual(quantized.shape, target_output.shape)
        self.assertEqual(
            len(self.adapter.attention_manifest_rows("JIQ_Joint_W4A4")), 2)
        self.assertEqual(
            len(self.adapter.concat_manifest_rows("JIQ_Joint_W4A4")), 1)
        self.assertEqual(
            self.adapter.attention_metric_rows("JIQ_Joint_W4A4")[0][
                "updates"], 1)
        self.assertEqual(
            self.adapter.concat_metric_rows("JIQ_Joint_W4A4")[0][
                "updates"], 1)
        self.assertGreater(len(
            self.adapter.search_rows("JIQ_Joint_W4A4")), 0)

    def test_reconstruction_rejects_more_forwards_than_target_pass(self):
        self.adapter.capture_targets()
        self.model(*self.inputs)
        self.adapter.observe_reconstruction()
        self.model(*self.inputs)

        with self.assertRaisesRegex(RuntimeError, "target forwards"):
            self.model(*self.inputs)

    def test_freeze_requires_every_target_to_be_consumed(self):
        self.adapter.capture_targets()
        self.model(*self.inputs)
        self.model(*self.inputs)
        self.adapter.observe_reconstruction()
        self.model(*self.inputs)

        with self.assertRaisesRegex(RuntimeError, "not fully consumed"):
            self.adapter.freeze()


class CompletionFormerJointQDropTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.model = ToyCompletionFormer().eval()
        self.adapter = make_adapter(self.model)
        self.addCleanup(self.adapter.close)
        self.inputs = make_inputs()
        self.owner = "backbone.former.block1.0"
        attention = self.owner + ".attn"
        concat = self.owner + ".concat_conv"
        self.sites = tuple([
            QDropActivationSite(
                site="attention::%s::%s" % (attention, role),
                owner_name=self.owner,
                owner_kind="attention_qkv",
                role="attention_%s" % role,
                signed=True,
                symmetric=True)
            for role in ("q", "k", "v")
        ] + [
            QDropActivationSite(
                site="concat::%s::%s_input" % (concat, role),
                owner_name=self.owner,
                owner_kind="concat_input",
                role="concat_%s_input" % role,
                signed=True,
                symmetric=True)
            for role in ("transformer", "cnn")
        ])
        self.quantizers = {}
        for index, site in enumerate(self.sites):
            quantizer = QDropActivationQuantizer(
                site=site.site,
                bits=4,
                signed=site.signed,
                symmetric=site.symmetric,
                scale_minimum=1.0e-8,
                seed=100 + index)
            quantizer.initialize(torch.tensor((-2.0, 2.0)))
            self.quantizers[site.site] = quantizer

    def test_external_qdrop_sites_consume_targets_and_freeze_deterministically(self):
        self.adapter.bind_qdrop_sites(self.sites, self.quantizers)
        self.assertEqual(len(tuple(
            self.adapter.qdrop_parameters(self.owner))), 5)

        self.adapter.capture_targets()
        reference = self.model(*self.inputs)
        for quantizer in self.quantizers.values():
            quantizer.start_reconstruction(quant_probability=1.0)
        self.adapter.observe_reconstruction()
        candidate = self.model(*self.inputs)

        self.assertEqual(candidate.shape, reference.shape)
        self.assertTrue(bool(torch.isfinite(candidate).all().item()))
        for quantizer in self.quantizers.values():
            self.assertEqual(quantizer.statistics()["calls"], 1)
        self.adapter.freeze_qdrop_sites(self.owner)
        self.adapter.freeze()
        for quantizer in self.quantizers.values():
            self.assertEqual(quantizer.phase, "frozen")

        first = self.model(*self.inputs)
        second = self.model(*self.inputs)
        torch.testing.assert_close(first, second)
        self.adapter.unbind_qdrop_sites()
        torch.testing.assert_close(self.model(*self.inputs), reference)

    def test_probability_tensor_cannot_be_bound_as_qdrop_site(self):
        probability = QDropActivationSite(
            site="signal::attention_probability",
            owner_name=self.owner,
            owner_kind="attention_probability",
            role="attention_probability",
            signed=False,
            symmetric=False)
        quantizer = self.quantizers[self.sites[0].site]

        with self.assertRaisesRegex(ValueError, "attention probability"):
            self.adapter.bind_qdrop_sites(
                (probability,), {probability.site: quantizer})

    def test_incomplete_binding_fails_without_mutating_adapter(self):
        site = self.sites[0]
        with self.assertRaisesRegex(RuntimeError, "incomplete QDrop Attention"):
            self.adapter.bind_qdrop_sites(
                (site,), {site.site: self.quantizers[site.site]})

        self.adapter.bind_qdrop_sites(self.sites, self.quantizers)
        self.assertEqual(len(tuple(
            self.adapter.qdrop_parameters(self.owner))), 5)

    def test_qdrop_execution_can_be_disabled_for_fp32_baseline(self):
        with torch.no_grad():
            reference = self.model(*self.inputs)
        for quantizer in self.quantizers.values():
            quantizer.start_reconstruction(quant_probability=1.0)
        self.adapter.bind_qdrop_sites(self.sites, self.quantizers)
        self.adapter.disable_qdrop_execution()

        with torch.no_grad():
            disabled = self.model(*self.inputs)
        torch.testing.assert_close(disabled, reference)

        self.adapter.enable_qdrop_execution()
        with torch.no_grad():
            enabled = self.model(*self.inputs)
        self.assertFalse(torch.equal(enabled, reference))

    def test_qdrop_range_calibration_does_not_cache_teacher_targets(self):
        self.adapter.observe_qdrop_ranges()
        output = self.model(*self.inputs)
        self.adapter.freeze_qdrop_ranges()

        self.assertTrue(bool(torch.isfinite(output).all().item()))
        metadata = self.adapter.calibration_metadata()
        self.assertEqual(metadata["qdrop_range_forwards"], 1)
        self.assertEqual(metadata["target_forwards"], 0)
        self.assertTrue(all(
            count == 0
            for count in metadata["attention_cached_samples"].values()))
        self.assertTrue(all(
            count == 0
            for count in metadata["concat_cached_samples"].values()))
        for site in self.sites:
            initialization = self.adapter.qdrop_initialization_tensor(site)
            self.assertTrue(bool(torch.isfinite(initialization).all().item()))
            self.assertGreater(float(initialization.abs().max().item()), 0.0)


class CompletionFormerJointHardDeploymentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = ToyCompletionFormer().eval()
        self.adapter = make_adapter(self.model)
        self.controller = make_hard_controller(self.model, self.adapter)

    def tearDown(self):
        if self.controller.installed:
            self.controller.remove()
        self.adapter.close()

    def test_frozen_quantizers_expose_joint_contract_and_codes(self):
        self.controller.install()

        for owner, quantizer in self.controller.activation_by_owner.items():
            self.assertEqual(quantizer.site, owner[0])
            self.assertEqual(quantizer.bits, 4)
            self.assertTrue(quantizer.signed)
            self.assertFalse(quantizer.unsigned)
            self.assertEqual((quantizer.qmin, quantizer.qmax), (-8, 7))
            output, codes = quantizer.quantize_with_codes(
                torch.tensor([-3.0, -0.1, 0.1, 3.0]))
            self.assertEqual(output.shape, codes.shape)
            self.assertEqual(codes.dtype, torch.int8)
            self.assertTrue(bool(torch.isfinite(output).all().item()))

    def test_hard_controller_executes_attention_and_concat_codes(self):
        self.controller.install()

        with torch.no_grad():
            output = self.model(*make_inputs())

        self.assertEqual(output.shape, (1, 8, 2, 2))
        self.assertTrue(bool(torch.isfinite(output).all().item()))


if __name__ == "__main__":
    unittest.main()
