"""Sage-only tests: gram_schmidt accelerator and Sage input handling."""
import unittest
from fractions import Fraction

import numpy as np
from sage.all import matrix, ZZ

from blaster._exact import rational_gso, to_int_matrix
from blaster.api import reduce_basis
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


if __name__ == "__main__":
    unittest.main()
