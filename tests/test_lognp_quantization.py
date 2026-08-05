import unittest

import torch

from scripts.lognp_quantization import (
    ChannelLogNPObserver,
    LogNPActivationQuantizer,
    LogNPQuantizationStats,
    fit_bias_correction,
    fit_weight_correction,
    lognp_inverse,
    lognp_transform,
)


class LogNPPrimitiveTest(unittest.TestCase):
    def test_round_trip_is_identity_before_quantization(self):
        values = torch.tensor([-100.0, -1.0, 0.0, 0.25, 10.0])
        alpha = torch.tensor(0.5)

        transformed = lognp_transform(values, alpha)

        torch.testing.assert_close(lognp_inverse(transformed, alpha), values)

    def test_signed_a4_uses_minus7_to_plus7_codes(self):
        quantizer = LogNPActivationQuantizer(
            bits=4,
            alpha=torch.tensor([1.0]),
            scale=torch.tensor([1.0]),
            unsigned=False,
        )

        _, codes = quantizer.quantize_with_codes(
            torch.tensor([[-100.0, -1.0, 0.0, 1.0, 100.0]]))

        self.assertEqual((int(codes.min()), int(codes.max())), (-7, 7))

    def test_unsigned_relu_a4_uses_zero_to15_codes(self):
        quantizer = LogNPActivationQuantizer(
            bits=4,
            alpha=torch.tensor([1.0]),
            scale=torch.tensor([1.0]),
            unsigned=True,
        )

        _, codes = quantizer.quantize_with_codes(
            torch.tensor([[0.0, 1.0, 1e20]]))

        self.assertEqual((int(codes.min()), int(codes.max())), (0, 15))

    def test_zero_is_preserved_and_inverse_stays_finite(self):
        quantizer = LogNPActivationQuantizer(
            bits=4,
            alpha=torch.tensor([0.25, 2.0]),
            scale=torch.tensor([0.5, 0.5]),
            unsigned=False,
        )
        values = torch.tensor([[[[0.0]], [[0.0]]]])

        reconstructed, _ = quantizer.quantize_with_codes(values)

        torch.testing.assert_close(reconstructed, values)
        self.assertTrue(bool(torch.isfinite(reconstructed).all()))

    def test_extreme_values_are_finite_and_monotonic(self):
        quantizer = LogNPActivationQuantizer(
            bits=8,
            alpha=torch.tensor([1.0]),
            scale=torch.tensor([0.25]),
            unsigned=True,
        )
        values = torch.tensor([[0.0, 1.0, 1e10, 1e20]])

        reconstructed, _ = quantizer.quantize_with_codes(values)

        self.assertTrue(bool(torch.isfinite(reconstructed).all()))
        self.assertTrue(bool(torch.diff(reconstructed, dim=1).ge(0).all()))

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            lognp_transform(torch.ones(2), torch.tensor(0.0))
        with self.assertRaises(ValueError):
            lognp_inverse(torch.ones(2), torch.tensor(float("nan")))


class ChannelLogNPObserverTest(unittest.TestCase):
    def test_calibration_is_per_channel_and_permutation_equivariant(self):
        values = torch.tensor([
            [[[1.0, 2.0]], [[10.0, 20.0]], [[100.0, 200.0]]],
            [[[2.0, 3.0]], [[20.0, 30.0]], [[200.0, 300.0]]],
        ])
        observer = ChannelLogNPObserver(sample_limit=16)
        observer.update(values)
        observer.freeze(bits=4, unsigned=False, alpha_factor=1.0)

        self.assertEqual(tuple(observer.alpha.shape), (3,))
        self.assertEqual(tuple(observer.scale.shape), (3,))
        self.assertTrue(bool(observer.alpha[0] < observer.alpha[1]))
        self.assertTrue(bool(observer.alpha[1] < observer.alpha[2]))

        permuted = ChannelLogNPObserver(sample_limit=16)
        permuted.update(values[:, [2, 0, 1]])
        permuted.freeze(bits=4, unsigned=False, alpha_factor=1.0)
        torch.testing.assert_close(
            permuted.alpha, observer.alpha[[2, 0, 1]])
        torch.testing.assert_close(
            permuted.scale, observer.scale[[2, 0, 1]])

    def test_repeated_updates_are_deterministic_and_bounded(self):
        values = torch.arange(2 * 2 * 8 * 8, dtype=torch.float32).reshape(
            2, 2, 8, 8)
        first = ChannelLogNPObserver(sample_limit=7)
        second = ChannelLogNPObserver(sample_limit=7)
        for _ in range(3):
            first.update(values)
            second.update(values)
        first.freeze(bits=4, unsigned=True, alpha_factor=0.5)
        second.freeze(bits=4, unsigned=True, alpha_factor=0.5)

        self.assertTrue(first.observed)
        self.assertEqual(first.sample_count, second.sample_count)
        torch.testing.assert_close(first.alpha, second.alpha)
        torch.testing.assert_close(first.scale, second.scale)
        self.assertTrue(all(item.numel() <= 7 for item in first.samples))

    def test_zero_channel_uses_finite_neutral_parameters(self):
        observer = ChannelLogNPObserver()
        observer.update(torch.zeros(1, 2, 3, 3))
        observer.freeze(bits=4, unsigned=True)

        torch.testing.assert_close(observer.alpha, torch.ones(2))
        torch.testing.assert_close(observer.scale, torch.ones(2))
        self.assertTrue(bool(torch.isfinite(observer.alpha).all()))
        self.assertTrue(bool(torch.isfinite(observer.scale).all()))

    def test_freeze_can_share_one_alpha_and_scale_across_channels(self):
        observer = ChannelLogNPObserver(sample_limit=32)
        observer.update(torch.tensor([
            [[[1.0, 2.0]], [[10.0, 20.0]]],
        ]))
        observer.freeze(bits=4, unsigned=False, per_channel=False)

        self.assertEqual(tuple(observer.alpha.shape), (1,))
        self.assertEqual(tuple(observer.scale.shape), (1,))
        reconstructed = observer.quantizer()(torch.tensor(
            [[[[1.0]], [[10.0]]]]))
        self.assertEqual(tuple(reconstructed.shape), (1, 2, 1, 1))


class LogNPQuantizationStatsTest(unittest.TestCase):
    def test_stats_report_tail_zero_clipping_and_nonfinite_counts(self):
        stats = LogNPQuantizationStats(sample_limit=16)
        reference = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        quantized = torch.tensor([-2.0, 0.0, 0.0, 1.0, float("nan")])
        codes = torch.tensor([-7, 0, 0, 7, 7], dtype=torch.int32)
        transformed = reference.abs()
        transformed_quantized = quantized.nan_to_num().abs()

        stats.update(
            reference,
            quantized,
            codes=codes,
            qmin=-7,
            qmax=7,
            transformed_reference=transformed,
            transformed_quantized=transformed_quantized,
        )

        self.assertEqual(stats.numel, 5)
        self.assertEqual(stats.nonfinite, 1)
        self.assertEqual(stats.zero_codes, 2)
        self.assertEqual(stats.saturated, 3)
        self.assertTrue(stats.p99_9 >= stats.p99 >= stats.p75)
        self.assertTrue(torch.isfinite(torch.tensor(stats.sqnr_db)))
        self.assertTrue(torch.isfinite(torch.tensor(stats.transformed_sqnr_db)))


class LogNPCompensationTest(unittest.TestCase):
    def test_bias_correction_removes_mean_output_residual(self):
        bias = torch.tensor([0.5, -0.25])
        reconstructed = torch.zeros(4, 2)
        target = torch.tensor([
            [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])

        corrected = fit_bias_correction(target, reconstructed, bias)

        torch.testing.assert_close(corrected, torch.tensor([1.5, -1.25]))
        residual = target - (reconstructed + corrected - bias)
        torch.testing.assert_close(residual.mean(dim=0), torch.zeros(2))

    def test_weight_correction_reduces_error_from_reconstructed_inputs(self):
        torch.manual_seed(21)
        original_inputs = torch.randn(64, 3)
        reconstructed_inputs = original_inputs + 0.15 * torch.randn(64, 3)
        target_weight = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]])
        target = torch.matmul(original_inputs, target_weight.t())

        corrected_weight = fit_weight_correction(
            reconstructed_inputs, target, ridge=1e-4)
        baseline_error = torch.mean(
            (torch.matmul(reconstructed_inputs, target_weight.t()) - target) ** 2)
        corrected_error = torch.mean(
            (torch.matmul(reconstructed_inputs, corrected_weight.t()) - target) ** 2)

        self.assertEqual(tuple(corrected_weight.shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(corrected_weight).all()))
        self.assertLess(float(corrected_error), float(baseline_error))


if __name__ == "__main__":
    unittest.main()
