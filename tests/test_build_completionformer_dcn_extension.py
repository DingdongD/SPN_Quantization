import unittest

from scripts.build_completionformer_dcn_extension import (
    replace_exact,
    transform_modulated_cuda,
)


class CompletionFormerDCNBuildTest(unittest.TestCase):
    def test_replace_exact_requires_declared_source_shape(self):
        with self.assertRaisesRegex(RuntimeError, "expected 2 occurrences"):
            replace_exact("old", "old", "new", expected_count=2)

    def test_transform_updates_pytorch_scalar_and_cuda_apis(self):
        source = "\n".join([
            "input.type().is_cuda()",
            "weight.type().is_cuda()",
            "bias.type().is_cuda()",
            "offset.type().is_cuda()",
            "mask.type().is_cuda()",
            "input.type().is_cuda()",
            "weight.type().is_cuda()",
            "bias.type().is_cuda()",
            "offset.type().is_cuda()",
            "mask.type().is_cuda()",
            "AT_DISPATCH_FLOATING_TYPES(input.type(), forward, body)",
            "AT_DISPATCH_FLOATING_TYPES(input.type(), backward, body)",
        ])

        transformed = transform_modulated_cuda(source)

        self.assertNotIn(".type().is_cuda()", transformed)
        self.assertEqual(transformed.count("input.scalar_type()"), 2)


if __name__ == "__main__":
    unittest.main()
