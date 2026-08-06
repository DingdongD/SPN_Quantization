import warnings
import unittest

import numpy as np
from PIL import Image

import data_transform


class DataTransformTest(unittest.TestCase):
    def test_float_pil_tensor_conversion_uses_writable_storage(self):
        image = Image.fromarray(np.ones((3, 4), dtype=np.float32), mode="F")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            tensor = data_transform.to_tensor(image)

        self.assertEqual(tuple(tensor.shape), (1, 3, 4))
        self.assertFalse(any("not writable" in str(item.message)
                             for item in captured))


if __name__ == "__main__":
    unittest.main()
