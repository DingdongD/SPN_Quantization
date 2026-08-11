import unittest
import json

import pytest
import torch
import torch.nn as nn

from scripts.run_nyu_edge_quantization import (
    load_reconstruction_manifest,
    normalize_merge_manifest,
    parse_edge_args,
)
from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
)
from spn_quant.deployment_contract import export_rounding_contracts
from spn_quant.qdrop_activation import QDropActivationQuantizer
from spn_quant.qdrop_contract import (
    build_qdrop_contract,
    save_qdrop_contract,
)
from spn_quant.qdrop_targets import (
    EXCLUDED_PROPAGATION_SITES,
    QDropActivationSite,
    QDropTargetPlan,
)


class EdgeRunnerTest(unittest.TestCase):
    def test_default_policy_preserves_shared_scale_baseline(self):
        options, remaining = parse_edge_args(["--run-dir", "run"])
        self.assertEqual(options.merge_policy, "shared")
        self.assertEqual(remaining, ["--run-dir", "run"])

    def test_grouped_policy_requires_positive_group_size(self):
        with self.assertRaises(SystemExit):
            parse_edge_args(["--merge-policy", "grouped"])
        options, remaining = parse_edge_args([
            "--merge-policy", "grouped", "--merge-group-size", "16",
            "--quant-backend", "hardware",
        ])
        self.assertEqual(options.merge_group_size, 16)
        self.assertEqual(remaining, ["--quant-backend", "hardware"])

    def test_group_size_is_rejected_for_non_grouped_policy(self):
        with self.assertRaises(SystemExit):
            parse_edge_args([
                "--merge-policy", "independent", "--merge-group-size", "16"])

    def test_manifest_normalization_preserves_legacy_columns(self):
        row = normalize_merge_manifest([{
            "merge": "cat#0", "policy": "independent", "scales": "0.1;1.0"
        }])[0]
        self.assertEqual(row["scale"], "")
        self.assertEqual(row["scales"], "0.1;1.0")


def write_qdrop_manifest(tmp_path):
    torch.manual_seed(53)
    model = nn.Sequential(nn.Linear(3, 2))
    rounding = AdaptiveRoundingController(
        model, AdaptiveRoundingConfig(bits=4))
    rounding.install(("0",))
    weights = export_rounding_contracts(rounding)
    site = QDropActivationSite(
        site="activation::0::input",
        owner_name="0",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("0",),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    activation = QDropActivationQuantizer(
        site=site.site,
        bits=4,
        signed=True,
        symmetric=True,
        scale_minimum=1.0e-8,
        seed=59,
    )
    activation.initialize(torch.tensor((-1.0, 1.0)))
    activation.start_reconstruction(1.0)
    activation.freeze()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"net": model.state_dict()}, checkpoint)
    contract = build_qdrop_contract(
        source_checkpoint=checkpoint,
        graph_contract={
            "fold": 1,
            "folded_pairs": [],
            "unfolded_fanout_pairs": [],
            "unfolded_conv_bn_pairs": [],
        },
        weight_contracts=weights,
        activation_contracts={site.site: activation.contract()},
        targets=plan,
        metadata={"seed": 59},
    )
    contract_path = save_qdrop_contract(
        tmp_path / "qdrop_contract.pt", contract)
    payload = {
        "format_version": 2,
        "strict": 1,
        "method": "qdrop_strict",
        "model": "cspn",
        "deployment_contract": str(contract_path),
        "targets": ["0"],
        "weight_bits": 4,
        "activation_bits": 4,
        "activation_policy": "exact_semantic_edge_contract",
    }
    manifest = tmp_path / "qdrop_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def test_loads_exact_qdrop_manifest_without_activation_recalibration(tmp_path):
    manifest, _ = write_qdrop_manifest(tmp_path)

    loaded = load_reconstruction_manifest(str(manifest))

    assert loaded["method"] == "qdrop_strict"
    assert loaded["weight_bits"] == 4
    assert loaded["activation_bits"] == 4
    assert loaded["activation_policy"] == "exact_semantic_edge_contract"
    assert loaded["qdrop_contract"]["format_version"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("weight_bits", 8),
        ("activation_bits", 8),
        ("activation_policy", "evaluation_backend_owned"),
        ("activation_manifest", [{"site": "activation::0::input"}]),
        ("activation_mode", "e2m1"),
    ),
)
def test_rejects_non_exact_qdrop_manifest(tmp_path, field, value):
    manifest, payload = write_qdrop_manifest(tmp_path)
    payload[field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((KeyError, ValueError)):
        load_reconstruction_manifest(str(manifest))


if __name__ == "__main__":
    unittest.main()
