"""Exact arithmetic primitives for the Stage 1 verifier and reference reducer."""
import unittest
from fractions import Fraction

import numpy as np

from blaster._exact import (
    coerce_int,
    column,
    delta_to_fraction,
    det_bareiss,
    full_column_rank,
    is_size_reduced,
    matmul_columns,
    nearest_core,
    nearest_panel,
    rational_gso,
    to_int_matrix,
    weak_lll_holds,
)


class TestCoerceInt(unittest.TestCase):
    def test_accepts_python_int(self):
        self.assertEqual(coerce_int(2**100), 2**100)

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            coerce_int(1.5)
        with self.assertRaises(TypeError):
            coerce_int(1.0)

    def test_rejects_bool(self):
        with self.assertRaises(TypeError):
            coerce_int(True)

    def test_rejects_fraction(self):
        with self.assertRaises(TypeError):
            coerce_int(Fraction(1, 2))


class TestToIntMatrix(unittest.TestCase):
    def test_nested_int_list(self):
        self.assertEqual(to_int_matrix([[1, 2], [3, 4]]), [[1, 2], [3, 4]])

    def test_numpy_int64(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        self.assertEqual(to_int_matrix(a), [[1, 2], [3, 4]])

    def test_large_entries_exceed_int64(self):
        rows = [[1, 2**60], [2**90, -1]]
        self.assertEqual(to_int_matrix(rows), rows)

    def test_rejects_float_entries(self):
        with self.assertRaises(TypeError):
            to_int_matrix([[1.5, 2]])

    def test_rejects_ragged(self):
        with self.assertRaises(ValueError):
            to_int_matrix([[1, 2], [3]])

    def test_rejects_flat_list(self):
        with self.assertRaises(ValueError):
            to_int_matrix([1, 2, 3])


class TestDetBareiss(unittest.TestCase):
    def test_2x2(self):
        self.assertEqual(det_bareiss([[1, 2], [3, 4]]), -2)

    def test_identity(self):
        self.assertEqual(det_bareiss([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1)

    def test_singular(self):
        self.assertEqual(det_bareiss([[1, 2], [2, 4]]), 0)

    def test_permutation_odd(self):
        self.assertEqual(det_bareiss([[0, 1], [1, 0]]), -1)

    def test_known_3x3(self):
        m = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        self.assertEqual(det_bareiss(m), 4)


class TestFullColumnRank(unittest.TestCase):
    def test_full_rank_diagonal(self):
        self.assertTrue(full_column_rank([[1, 0], [0, 1]]))

    def test_dependent_columns(self):
        # columns (1,2) and (2,4) are proportional
        self.assertFalse(full_column_rank([[1, 2], [2, 4]]))

    def test_zero_column(self):
        self.assertFalse(full_column_rank([[1, 0], [0, 0]]))


class TestMatmulColumns(unittest.TestCase):
    def test_identity_transform(self):
        b0 = [[1, 2], [3, 4]]
        u = [[1, 0], [0, 1]]
        self.assertEqual(matmul_columns(b0, u), b0)

    def test_scaling_column(self):
        b0 = [[1, 2], [3, 4]]
        u = [[2, 0], [0, 1]]
        self.assertEqual(matmul_columns(b0, u), [[2, 2], [6, 4]])


class TestDeltaToFraction(unittest.TestCase):
    def test_099_is_binary64_dyadic(self):
        self.assertEqual(delta_to_fraction(0.99),
                         Fraction(*float(0.99).as_integer_ratio()))
        self.assertNotEqual(delta_to_fraction(0.99), Fraction(99, 100))

    def test_exact_dyadics_round_trip(self):
        self.assertEqual(delta_to_fraction(0.5), Fraction(1, 2))
        self.assertEqual(delta_to_fraction(0.75), Fraction(3, 4))

    def test_exact_fraction_delta_is_preserved(self):
        # An exact rational delta must not be rounded through binary64.
        self.assertEqual(delta_to_fraction(Fraction(99, 100)), Fraction(99, 100))
        self.assertEqual(delta_to_fraction(Fraction(3, 4)), Fraction(3, 4))

    def test_exact_fraction_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            delta_to_fraction(Fraction(1, 4))  # 0.25 is not strictly > 0.25
        with self.assertRaises(ValueError):
            delta_to_fraction(Fraction(1, 1))


class TestRationalGso(unittest.TestCase):
    def test_2x2(self):
        # columns (2,0) and (1,1): mu[0][1] = 1/2, r = [4, 1]
        mu, r = rational_gso([[2, 0], [1, 1]])
        self.assertEqual(mu[0][1], Fraction(1, 2))
        self.assertEqual(r, [Fraction(4), Fraction(1)])

    def test_identity(self):
        mu, r = rational_gso([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(r, [Fraction(1)] * 3)
        for i in range(3):
            for j in range(3):
                self.assertEqual(mu[i][j], Fraction(0))


class TestWeakLllAndSizeReduced(unittest.TestCase):
    def test_weak_lll_strict(self):
        # r = [4,1], mu[0][1]=1/2: 1 + 1/4*4 = 2 > delta*4 iff delta < 1/2
        mu, r = rational_gso([[2, 0], [1, 1]])
        self.assertTrue(weak_lll_holds(mu, r, Fraction(1, 4)))
        self.assertFalse(weak_lll_holds(mu, r, Fraction(1, 2)))

    def test_size_reduced(self):
        mu, _ = rational_gso([[2, 0], [1, 1]])  # mu = 1/2 is allowed
        self.assertTrue(is_size_reduced(mu))
        mu_bad, _ = rational_gso([[1, 0], [1, 1]])  # mu = 1 is not
        self.assertFalse(is_size_reduced(mu_bad))


class TestNearest(unittest.TestCase):
    def test_nearest_core_ties_away_from_zero(self):
        cases = {
            Fraction(1, 2): 1, Fraction(-1, 2): -1,
            Fraction(3, 2): 2, Fraction(-3, 2): -2,
            Fraction(7, 3): 2, Fraction(-7, 3): -2,
            Fraction(0): 0, Fraction(2): 2,
        }
        for x, want in cases.items():
            with self.subTest(x=x):
                self.assertEqual(nearest_core(x), want)

    def test_nearest_panel_rounds_half_to_even(self):
        cases = {
            Fraction(1, 2): 0, Fraction(-1, 2): 0,
            Fraction(3, 2): 2, Fraction(5, 2): 2, Fraction(7, 2): 4,
            Fraction(-3, 2): -2, Fraction(-5, 2): -2,
            Fraction(1, 3): 0, Fraction(2, 3): 1, Fraction(1): 1,
        }
        for x, want in cases.items():
            with self.subTest(x=x):
                self.assertEqual(nearest_panel(x), want)

    def test_agree_off_ties(self):
        for x in (Fraction(1, 3), Fraction(2, 3), Fraction(1, 4),
                  Fraction(3, 4), Fraction(5, 3), Fraction(-2, 3)):
            self.assertEqual(nearest_core(x), nearest_panel(x))


if __name__ == "__main__":
    unittest.main()
