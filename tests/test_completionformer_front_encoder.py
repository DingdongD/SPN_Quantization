import unittest

from spn_quant.completionformer_front_encoder import (
    FRONT_ENCODER_UNIT_ORDER,
    promotion_overrides,
    resolve_front_encoder_units,
    unit_manifest_rows,
)


def block_weight_names(stage, index, downsample=False):
    prefix = "backbone.former.embed_layer%d.%d" % (stage, index)
    names = [
        prefix + ".ca.fc.0",
        prefix + ".ca.fc.2",
        prefix + ".conv1",
        prefix + ".conv2",
        prefix + ".sa.conv1",
    ]
    if downsample:
        names.append(prefix + ".downsample.0")
    return names


def official_weight_names():
    names = [
        "backbone.conv1.0",
        "backbone.conv1_dep.0",
        "backbone.conv1_rgb.0",
    ]
    for index in range(3):
        names.extend(block_weight_names(1, index))
    for index in range(4):
        names.extend(block_weight_names(2, index, downsample=index == 0))
    names.append("backbone.former.patch_embed1.proj")
    return tuple(names)


def official_activation_sites():
    names = set(official_weight_names())
    names.update({
        "backbone.conv1.1#0",
        "backbone.conv1_dep.1#0",
        "backbone.conv1_rgb.1#0",
        "backbone.former.patch_embed1.norm",
    })
    for stage, count in ((1, 3), (2, 4)):
        for index in range(count):
            prefix = "backbone.former.embed_layer%d.%d" % (stage, index)
            names.update({
                prefix + ".relu#0",
                prefix + ".relu#1",
                prefix + ".ca.fc.1#0",
                prefix + ".ca.fc.1#1",
            })
    return tuple(sorted(names))


class CompletionFormerFrontEncoderUnitTest(unittest.TestCase):
    def test_resolve_units_preserves_official_atomic_order(self):
        units = resolve_front_encoder_units(
            official_weight_names(), official_activation_sites())

        self.assertEqual(tuple(units), FRONT_ENCODER_UNIT_ORDER)
        self.assertEqual(units["Stem"]["weight_modules"], (
            "backbone.conv1_rgb.0",
            "backbone.conv1_dep.0",
            "backbone.conv1.0",
        ))
        self.assertIn(
            "backbone.former.embed_layer2.0.downsample.0",
            units["Embed2.0"]["weight_modules"])
        self.assertEqual(units["PatchEmbed1"]["activation_sites"], (
            "backbone.former.patch_embed1.proj",
            "backbone.former.patch_embed1.norm",
        ))

    def test_missing_official_weight_module_fails(self):
        names = tuple(
            name for name in official_weight_names()
            if name != "backbone.former.embed_layer2.0.downsample.0")

        with self.assertRaisesRegex(
                ValueError, "front encoder weight modules mismatch"):
            resolve_front_encoder_units(names, official_activation_sites())

    def test_unexpected_matching_weight_module_fails(self):
        names = official_weight_names() + (
            "backbone.former.embed_layer1.0.unexpected",)

        with self.assertRaisesRegex(
                ValueError, "front encoder weight modules mismatch"):
            resolve_front_encoder_units(names, official_activation_sites())

    def test_missing_activation_site_fails(self):
        sites = tuple(
            name for name in official_activation_sites()
            if name != "backbone.former.embed_layer1.0.relu#1")

        with self.assertRaisesRegex(
                ValueError, "front encoder activation sites mismatch"):
            resolve_front_encoder_units(official_weight_names(), sites)

    def test_promotion_overrides_cover_selected_units_only(self):
        units = resolve_front_encoder_units(
            official_weight_names(), official_activation_sites())

        overrides = promotion_overrides(
            units, ("Stem", "Embed2.0"))

        self.assertTrue(all(
            bits == 8
            for bits in overrides["weight_bit_overrides"].values()))
        self.assertTrue(all(
            bits == 8
            for bits in overrides["activation_bit_overrides"].values()))
        self.assertIn(
            "backbone.former.embed_layer2.0.downsample.0",
            overrides["weight_bit_overrides"])
        self.assertIn(
            "backbone.conv1_dep.1#0",
            overrides["activation_bit_overrides"])
        self.assertNotIn(
            "backbone.former.embed_layer1.0.conv1",
            overrides["weight_bit_overrides"])

    def test_unknown_or_out_of_order_selection_fails(self):
        units = resolve_front_encoder_units(
            official_weight_names(), official_activation_sites())

        with self.assertRaisesRegex(ValueError, "unknown front encoder units"):
            promotion_overrides(units, ("Missing",))
        with self.assertRaisesRegex(ValueError, "official unit order"):
            promotion_overrides(units, ("Embed1.0", "Stem"))

    def test_manifest_has_unique_weight_ownership(self):
        units = resolve_front_encoder_units(
            official_weight_names(), official_activation_sites())

        rows = unit_manifest_rows(units)
        weight_rows = [row for row in rows if row["kind"] == "weight"]
        names = [row["site"] for row in weight_rows]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            [row["unit"] for row in rows if row["kind"] == "unit"],
            list(FRONT_ENCODER_UNIT_ORDER))


if __name__ == "__main__":
    unittest.main()

