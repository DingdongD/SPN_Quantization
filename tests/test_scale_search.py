import unittest

import torch

from spn_quant.scale_search import CalibrationCache, CoordinateScaleSearch


class CalibrationCacheTest(unittest.TestCase):
    def test_cache_detaches_and_preserves_insertion_order(self):
        cache = CalibrationCache(sample_limit=2, byte_limit=128)
        first = torch.tensor([1.0, 2.0], requires_grad=True)
        second = torch.tensor([3.0, 4.0])

        cache.append((first,))
        cache.append((second,))
        first.detach().zero_()

        samples = cache.samples()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0][0].device.type, "cpu")
        self.assertEqual(samples[0][0].dtype, torch.float32)
        self.assertEqual(samples[0][0].tolist(), [1.0, 2.0])
        self.assertEqual(samples[1][0].tolist(), [3.0, 4.0])

    def test_cache_rejects_sample_limit_instead_of_evicting(self):
        cache = CalibrationCache(sample_limit=1, byte_limit=128)
        cache.append((torch.ones(2),))

        with self.assertRaisesRegex(RuntimeError, "sample limit"):
            cache.append((torch.ones(2) * 2.0,))

        self.assertEqual(cache.samples()[0][0].tolist(), [1.0, 1.0])

    def test_cache_rejects_byte_limit(self):
        cache = CalibrationCache(sample_limit=2, byte_limit=16)
        cache.append((torch.ones(2),))

        with self.assertRaisesRegex(RuntimeError, "byte limit"):
            cache.append((torch.ones(4),))

    def test_cache_rejects_nonfinite_tensor(self):
        cache = CalibrationCache(sample_limit=1, byte_limit=128)

        with self.assertRaises(ValueError):
            cache.append((torch.tensor([float("nan")]),))


class CoordinateScaleSearchTest(unittest.TestCase):
    def test_search_is_deterministic(self):
        search = CoordinateScaleSearch(
            parameter_names=("q", "k"),
            factors=(1.0, 0.75, 0.5), rounds=2)

        result = search.run(
            {"q": 1.0, "k": 1.0},
            lambda values: (values["q"] - 0.75) ** 2 +
            (values["k"] - 0.5) ** 2,
            sample_count=4)

        self.assertEqual(result.values, {"q": 0.75, "k": 0.5})
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.rows[0]["parameter"], "q")
        self.assertEqual(result.rows[0]["round"], 0)
        self.assertEqual(result.rows[0]["sample_count"], 4)

    def test_search_keeps_current_value_on_exact_tie(self):
        search = CoordinateScaleSearch(
            parameter_names=("scale",), factors=(1.0, 0.5), rounds=2)

        result = search.run(
            {"scale": 2.0}, lambda values: 1.0, sample_count=1)

        self.assertEqual(result.values["scale"], 2.0)
        selected = [row for row in result.rows if row["selected"]]
        self.assertEqual([row["factor"] for row in selected], [1.0, 1.0])

    def test_search_rejects_missing_parameter(self):
        search = CoordinateScaleSearch(
            parameter_names=("q", "k"), factors=(1.0,), rounds=1)

        with self.assertRaises(KeyError):
            search.run({"q": 1.0}, lambda values: 0.0, sample_count=1)

    def test_search_rejects_nonfinite_objective(self):
        search = CoordinateScaleSearch(
            parameter_names=("q",), factors=(1.0,), rounds=1)

        with self.assertRaises(ValueError):
            search.run(
                {"q": 1.0}, lambda values: float("inf"), sample_count=1)


if __name__ == "__main__":
    unittest.main()
