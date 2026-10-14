"""Sage-only tests: gram_schmidt accelerator and Sage input handling."""
import unittest
from fractions import Fraction

import numpy as np
from sage.all import matrix, ZZ

from blaster._exact import rational_gso, to_int_matrix
from blaster.api import reduce_basis
from blaster.reference import lll_reduce, lll_reduce_rows
from blaster.verify import verify_reduction
from blaster.verify_sage import sage_gso


def _cols_of(B_rows):
    m = len(B_rows)
    return [[B_rows[r][j] for r in range(m)] for j in range(len(B_rows[0]))]


class TestSageGso(unittest.TestCase):
    def test_sage_gso_agrees_with_rational_gso(self):
        # Non-symmetric basis: a row/column confusion would disagree here.
        B_rows = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        cols = _cols_of(B_rows)
        mu_s, r_s = sage_gso(cols)
        mu_r, r_r = rational_gso(cols)
        for i in range(len(cols)):
            self.assertEqual(r_s[i], r_r[i])
            for j in range(len(cols)):
                self.assertEqual(mu_s[i][j], mu_r[i][j])

    def test_verify_agrees_with_either_backend(self):
        B0 = [[2, 0], [1, 1]]
        rep_rat = verify_reduction(B0, B0, [[1, 0], [0, 1]], delta=0.99)
        rep_sage = verify_reduction(B0, B0, [[1, 0], [0, 1]], delta=0.99, gso=sage_gso)
        self.assertEqual(rep_rat.verified, rep_sage.verified)
        self.assertEqual(rep_rat.weak_lll_ok, rep_sage.weak_lll_ok)
        self.assertEqual(rep_rat.size_reduced_ok, rep_sage.size_reduced_ok)

    def test_verify_gso_string_sage_matches_default(self):
        B0 = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        rep_rat = verify_reduction(B0, B0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], delta=0.99)
        rep_sage = verify_reduction(
            B0, B0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], delta=0.99, gso="sage"
        )
        self.assertEqual(rep_rat.verified, rep_sage.verified)
        self.assertEqual(rep_rat.weak_lll_ok, rep_sage.weak_lll_ok)
        self.assertEqual(rep_rat.size_reduced_ok, rep_sage.size_reduced_ok)

    def test_to_int_matrix_accepts_sage(self):
        self.assertEqual(to_int_matrix(matrix(ZZ, [[1, 2], [3, 4]])), [[1, 2], [3, 4]])


class TestReduceBasisSageType(unittest.TestCase):
    def test_reduce_basis_preserves_sage_matrix_type(self):
        A = matrix(ZZ, [[10, 0, 0], [1, 10, 0], [0, 1, 1]])
        B_red, T, _ = reduce_basis(A, return_transform=True, cores=1)
        self.assertTrue(type(B_red).__module__.startswith("sage."))
        self.assertTrue(type(T).__module__.startswith("sage."))

    def test_reference_engine_preserves_sage_matrix_type(self):
        A = matrix(ZZ, [[10, 0, 0], [1, 10, 0], [0, 1, 1]])
        B_red, T, _ = reduce_basis(A, engine="reference", return_transform=True)
        self.assertTrue(type(B_red).__module__.startswith("sage."))
        self.assertTrue(type(T).__module__.startswith("sage."))

    def test_reduce_basis_reference_gso_sage_matches_default(self):
        A = matrix(ZZ, [[10, 0, 0], [1, 10, 0], [0, 1, 1]])
        red_def, T_def, _ = reduce_basis(A, engine="reference", return_transform=True)
        red_sage, T_sage, _ = reduce_basis(
            A, engine="reference", gso="sage", return_transform=True
        )
        self.assertEqual(red_def, red_sage)
        self.assertEqual(T_def, T_sage)


class TestReferenceGsoBackend(unittest.TestCase):
    """The reference reducer's ``gso`` backend is a drop-in exact primitive."""

    def test_lll_reduce_gso_sage_matches_default(self):
        B0 = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        cols_def, U_def, tr_def = lll_reduce(B0, delta=0.99)
        cols_sage, U_sage, tr_sage = lll_reduce(B0, delta=0.99, gso="sage")
        self.assertEqual(cols_def, cols_sage)
        self.assertEqual(U_def, U_sage)
        self.assertEqual(tr_def, tr_sage)

    def test_lll_reduce_rows_gso_sage_matches_default(self):
        A0 = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        rows_def, T_def, tr_def = lll_reduce_rows(A0, delta=0.99)
        rows_sage, T_sage, tr_sage = lll_reduce_rows(A0, delta=0.99, gso="sage")
        self.assertEqual(rows_def, rows_sage)
        self.assertEqual(T_def, T_sage)
        self.assertEqual(tr_def, tr_sage)

    def test_lll_reduce_gso_invalid_selector_raises(self):
        with self.assertRaisesRegex(ValueError, "gso"):
            lll_reduce([[3, 1, 4], [1, 5, 9], [2, 6, 5]], gso="numpy")


if __name__ == "__main__":
    unittest.main()
