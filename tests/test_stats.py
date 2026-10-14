import unittest

import numpy as np

from blaster.stats import get_profile


class TestGetProfileExact(unittest.TestCase):
    def test_object_dtype_matches_int64(self):
        small = np.array([[7, 2], [0, 5]], dtype=np.int64)
        expected = get_profile(small)
        actual = get_profile(np.asarray(small, dtype=object))
        self.assertEqual(actual, expected)

    def test_arbitrary_precision_entries(self):
        big = 1 << 2000  # exceeds the float64 range that np.linalg.qr coerces to
        basis = np.array([[big, 0], [0, 1]], dtype=object)
        self.assertEqual(get_profile(basis), [2000.0, 0.0])


if __name__ == "__main__":
    unittest.main()
