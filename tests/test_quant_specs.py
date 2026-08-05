import unittest

from spn_quant.specs import QuantSpec


class QuantSpecTest(unittest.TestCase):
    def test_default_w4a4_contract_is_uniform_not_lognp(self):
        spec = QuantSpec.signed_tensor(4)
        self.assertEqual(spec.bits, 4)
        self.assertEqual(spec.transform, "none")
        self.assertEqual(spec.granularity, "tensor")
        self.assertTrue(spec.signed)

    def test_group_quantization_requires_axis_and_group_size(self):
        with self.assertRaises(ValueError):
            QuantSpec(bits=4, granularity="group", axis=1)
        spec = QuantSpec(bits=4, granularity="group", axis=1, group_size=32)
        self.assertEqual(spec.group_size, 32)

    def test_unsigned_contract_uses_affine_and_preserves_zero(self):
        spec = QuantSpec.unsigned_tensor(4)
        self.assertEqual(spec.scheme, "affine")
        self.assertFalse(spec.signed)
        self.assertTrue(spec.preserve_zero)

    def test_lognp_is_explicit_only(self):
        base = QuantSpec.signed_tensor(4)
        lognp = base.with_transform("lognp")
        self.assertEqual(base.transform, "none")
        self.assertEqual(lognp.transform, "lognp")

    def test_manifest_is_csv_friendly(self):
        spec = QuantSpec(bits=4, granularity="group", axis=1, group_size=16)
        row = spec.manifest()
        self.assertEqual(row["axis"], 1)
        self.assertEqual(row["group_size"], 16)


if __name__ == "__main__":
    unittest.main()
