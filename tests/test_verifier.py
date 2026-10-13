"""Structural checks of the Stage 1 independent verifier."""
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
