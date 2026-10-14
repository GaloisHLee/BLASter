import math
import unittest

import numpy as np

from blaster.stats import (
    gaussian_heuristic,
    get_profile,
    gh,
    potential,
    rhf,
    slope,
)


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


class TestStatsConsumersArbitraryPrecision(unittest.TestCase):
    # E-6: rhf/slope/potential/gaussian_heuristic must consume the exact
    # get_profile output (0.5*log2(r_i) values) produced for an arbitrary-
    # precision basis without regressing. The basis below is diagonal with
    # entries 2^2000, 2^1000, 1, so the exact GSO gives r = [2^4000, 2^2000, 1]
    # and the profile is [2000.0, 1000.0, 0.0]. Every consumer is checked
    # against a hand-derived closed form, not against the implementation.

    def setUp(self):
        big2000 = 1 << 2000
        big1000 = 1 << 1000
        basis = np.array(
            [[big2000, 0, 0], [0, big1000, 0], [0, 0, 1]], dtype=object
        )
        self.profile = get_profile(basis)
        self.assertEqual(self.profile, [2000.0, 1000.0, 0.0])

    def test_rhf_matches_closed_form(self):
        # rhf = 2**((p0 - sum/n)/n) = 2**((2000 - 1000)/3) = 2**(1000/3)
        self.assertEqual(rhf(self.profile), 2.0 ** (1000.0 / 3.0))
        self.assertTrue(math.isfinite(rhf(self.profile)))

    def test_slope_matches_closed_form(self):
        # i_mean = 1; v1 = 2000*(-1) + 1000*0 + 0*1 = -2000; v2 = 1+0+1 = 2
        self.assertEqual(slope(self.profile), -1000.0)

    def test_potential_matches_closed_form(self):
        # sum((n-i)*p_i) = 3*2000 + 2*1000 + 1*0 = 8000
        self.assertEqual(potential(self.profile), 8000.0)

    def test_gaussian_heuristic_matches_closed_form(self):
        # gh(n) * 2**(sum/n) = gh(3) * 2**(3000/3) = gh(3) * 2**1000
        self.assertEqual(gaussian_heuristic(self.profile), gh(3) * 2.0**1000)
        self.assertTrue(math.isfinite(gaussian_heuristic(self.profile)))


if __name__ == "__main__":
    unittest.main()
