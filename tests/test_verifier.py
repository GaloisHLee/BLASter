"""Structural checks of the Stage 1 independent verifier."""
import unittest

import numpy as np

from blaster import reduce
from blaster.verify import VerificationFailed, verify_reduction


I2 = [[1, 0], [0, 1]]


class TestStructuralChecks(unittest.TestCase):
    def test_certifies_identity(self):
        rep = verify_reduction(I2, I2, I2, delta=0.99)
        self.assertTrue(rep.verified)
        self.assertTrue(rep.structural_ok)

    def test_product_mismatch_raises(self):
        B = [[1, 0], [0, 2]]
        with self.assertRaises(VerificationFailed):
            verify_reduction(I2, B, I2, delta=0.99)

    def test_nonunimodular_raises(self):
        U = [[2, 0], [0, 1]]
        B = matmul(I2, U)
        with self.assertRaises(VerificationFailed) as cm:
            verify_reduction(I2, B, U, delta=0.99)
        self.assertIn("det", str(cm.exception))

    def test_rank_deficient_raises(self):
        B0 = [[1, 2], [2, 4]]  # columns (1,2),(2,4) dependent
        with self.assertRaises(VerificationFailed):
            verify_reduction(B0, B0, I2, delta=0.99)

    def test_shape_mismatch_raises(self):
        B = [[1, 0, 0], [0, 1, 0]]
        with self.assertRaises(VerificationFailed):
            verify_reduction(I2, B, I2, delta=0.99)

    def test_delta_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            verify_reduction(I2, I2, I2, delta=1.0)

    def test_accepts_numpy_inputs(self):
        a = np.array([[10, 0, 0], [1, 10, 0], [0, 1, 1]], dtype=np.int64)
        rep = verify_reduction(a, a, np.eye(3, dtype=np.int64), delta=0.99)
        self.assertTrue(rep.structural_ok)


def matmul(A, U):
    n = len(A[0])
    p = len(U[0])
    out = [[0] * p for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(p):
            out[i][j] = sum(A[i][k] * U[k][j] for k in range(n))
    return out


def diag(*xs):
    n = len(xs)
    return [[xs[i] if i == j else 0 for j in range(n)] for i in range(n)]


class TestPostconditions(unittest.TestCase):
    def test_size_reduction_corruption_reported_not_raised(self):
        # row-major [[1,1],[0,1]] means columns (1,0),(1,1): mu[0][1] = 1 > 1/2
        B0 = [[1, 1], [0, 1]]
        rep = verify_reduction(B0, B0, I2, delta=0.99)
        self.assertFalse(rep.size_reduced_ok)
        self.assertTrue(rep.weak_lll_ok)
        self.assertFalse(rep.verified)

    def test_lovasz_equality_boundary_is_strict(self):
        # row-major [[2,1],[0,1]] means columns (2,0),(1,1): r0=4, r1=1, mu=1/2
        # r1 + mu^2 r0 = 1 + 1 = 2 == delta*r0 = 0.5*4 = 2 -> strict fails
        B0 = [[2, 1], [0, 1]]
        rep = verify_reduction(B0, B0, I2, delta=0.5)
        self.assertFalse(rep.weak_lll_ok)
        self.assertTrue(rep.size_reduced_ok)  # mu = 1/2 is allowed
        self.assertFalse(rep.verified)

    def test_wide_dynamic_range_certified(self):
        B0 = diag(1, 2**30, 2**60, 2**90)
        rep = verify_reduction(B0, B0, diag(1, 1, 1, 1), delta=0.99)
        self.assertTrue(rep.verified)

    def test_delta_reflected_in_report(self):
        rep = verify_reduction(I2, I2, I2, delta=0.99)
        from fractions import Fraction
        self.assertEqual(rep.delta, Fraction(*float(0.99).as_integer_ratio()))


class TestM0Corpus(unittest.TestCase):
    """Acceptance #1: certify the current FastInt64 output on the M0 corpus."""

    CASES = {
        "diag3": np.diag([10, 10, 10]),
        "ex3": np.array([[10, 0, 0], [1, 10, 0], [0, 1, 1]]),
        "ex3b": np.array([[12, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "rand4": np.array([[5, 2, 8, 3], [7, 1, 4, 9], [2, 6, 3, 8], [4, 9, 1, 7]]),
        "rand5": np.array(
            [[3, 7, 1, 9, 2], [8, 4, 6, 5, 1], [9, 2, 7, 4, 8],
             [1, 5, 3, 8, 6], [4, 8, 2, 6, 7]]
        ),
        "rand6": np.random.RandomState(0).randint(-20, 20, size=(6, 6)).astype(np.int64),
    }

    def test_fastint64_output_is_certified(self):
        for name, A in self.CASES.items():
            with self.subTest(name=name):
                n = A.shape[1]
                U, B, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                rep = verify_reduction(A, B, U, delta=0.99)
                self.assertTrue(rep.structural_ok, name)
                self.assertTrue(rep.weak_lll_ok, name)
                self.assertTrue(rep.size_reduced_ok, name)
                self.assertTrue(rep.verified, name)


class TestGsoSelector(unittest.TestCase):
    """verify_reduction's ``gso`` argument selects the exact GSO backend."""

    def test_gso_string_rational_matches_default(self):
        rep_default = verify_reduction(I2, I2, I2, delta=0.99)
        rep_rational = verify_reduction(I2, I2, I2, delta=0.99, gso="rational")
        self.assertEqual(rep_default.verified, rep_rational.verified)
        self.assertEqual(rep_default.weak_lll_ok, rep_rational.weak_lll_ok)
        self.assertEqual(rep_default.size_reduced_ok, rep_rational.size_reduced_ok)

    def test_gso_callable_still_supported(self):
        from blaster._exact import rational_gso
        rep = verify_reduction(I2, I2, I2, delta=0.99, gso=rational_gso)
        self.assertTrue(rep.verified)

    def test_gso_invalid_selector_raises(self):
        with self.assertRaisesRegex(ValueError, "gso"):
            verify_reduction(I2, I2, I2, delta=0.99, gso="numpy")


if __name__ == "__main__":
    unittest.main()
