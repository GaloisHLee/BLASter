import math
import unittest
from fractions import Fraction

from blaster._bkz import (
    PRUNING_PARAMS,
    _gh_squared,
    _safe_gh_squared,
    get_pruning_coefficients,
)


class TestPruningParams(unittest.TestCase):
    def test_table_has_41_rows_and_2k_lengths(self):
        self.assertEqual(len(PRUNING_PARAMS), 41)
        for k in range(1, 41):
            self.assertEqual(len(PRUNING_PARAMS[k]), 2 * k)

    def test_bkz2_coefficients(self):
        self.assertEqual(
            get_pruning_coefficients(2),
            [Fraction(1, 1), Fraction(1, 10000)],
        )

    def test_odd_block_slices_next_even_row(self):
        # block 3 -> index (3+1)//2 = 2 (BKZ-4 row), first 3 of its 4 entries
        self.assertEqual(
            get_pruning_coefficients(3),
            [Fraction(1, 1), Fraction(7347, 10000), Fraction(4695, 10000)],
        )

    def test_block_size_range_and_type(self):
        for bad in (1, 81, 3.5, True):
            with self.assertRaises(ValueError):
                get_pruning_coefficients(bad)

    def test_gh_squared_dim2_volume_zero(self):
        # lgamma(2.0) == 0, so _gh_squared(2, 0.0) == exp(0)/pi == 1/pi
        self.assertEqual(_gh_squared(2, 0.0), 1.0 / math.pi)

    def test_safe_gh_squared_returns_exact_fraction(self):
        v = _safe_gh_squared(20, math.log(2.0))
        self.assertIsInstance(v, Fraction)
        self.assertGreater(v, 0)


if __name__ == "__main__":
    unittest.main()
