"""Reference reducer: exact BLASter transcription for ordinary LLL."""
import unittest

import numpy as np

from blaster import reduce, verify_reduction
from blaster.reference import lll_reduce


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


class TestReferenceMatchesFastInt64(unittest.TestCase):
    """Acceptance #5: exact B/U equality pins the same deterministic decisions."""

    def test_deterministic_cases_match(self):
        for name, A in CASES.items():
            with self.subTest(name=name):
                n = A.shape[1]
                U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)

    def test_fuzz_matches(self):
        rs = np.random.RandomState(1234)
        for n in range(2, 9):
            for trial in range(20):
                A = rs.randint(-30, 30, size=(n, n)).astype(np.int64)
                try:
                    U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                except Exception:
                    continue  # rank-deficient or non-square reduce rejected
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)


class TestReferenceOutputAcceptedByVerifier(unittest.TestCase):
    """Acceptance #4: the reference output is certified by the independent verifier."""

    def test_output_verified(self):
        for name, A in CASES.items():
            with self.subTest(name=name):
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                rep = verify_reduction(A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99)
                self.assertTrue(rep.verified, name)

    def test_hnp_basis_certified(self):
        """Acceptance #3: a 61-bit HNP lattice beyond the float64 exact range.

        Findings #2/#3 record float64 rank false-negatives and size-reduction
        errors on 60+ bit HNP lattices. Entries here reach ~2^61 and Gram entries
        ~2^122, so the float fast path (`reduce`) cannot certify them; the exact
        reference reducer plus exact verifier can.
        """
        p = (1 << 61) - 1  # Mersenne prime
        t = [int(x) for x in np.random.RandomState(7).randint(0, p, size=3)]
        B0 = [
            [p, 0, 0, 0],
            [0, p, 0, 0],
            [0, 0, p, 0],
            [t[0], t[1], t[2], 1],
        ]
        cols_ref, U_ref, _ = lll_reduce(B0, delta=0.99)
        B_ref = np.array(cols_ref, dtype=object).T
        rep = verify_reduction(B0, B_ref, np.array(U_ref, dtype=object), delta=0.99)
        self.assertTrue(rep.verified)


class TestTranscript(unittest.TestCase):
    def test_transcript_replays_to_same_result(self):
        A = CASES["rand4"]
        n = A.shape[1]
        rows = A.tolist()
        B = [[rows[r][j] for r in range(n)] for j in range(n)]
        U = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
        cols_ref, U_ref, tr = lll_reduce(A, delta=0.99)
        for rec in tr:
            if rec[0] == "add":
                _, i, k, z, _ = rec
                B[k] = [a + z * b for a, b in zip(B[k], B[i])]
                U[k] = [a + z * b for a, b in zip(U[k], U[i])]
            elif rec[0] == "swap":
                _, a, b = rec
                B[a], B[b] = B[b], B[a]
                U[a], U[b] = U[b], U[a]
            # "offset" records carry no column operation
        self.assertEqual(B, cols_ref)
        # replay accumulates columns; U_ref is row-major, so transpose to compare
        U_rows = [[U[j][r] for j in range(n)] for r in range(n)]
        self.assertEqual(U_rows, U_ref)

    def test_transcript_shape(self):
        cols_ref, U_ref, tr = lll_reduce(CASES["ex3"], delta=0.99)
        self.assertEqual(tr[0][0], "offset")
        self.assertEqual(tr[0][1], 1)  # floor(3/2) = 1 first iteration
        self.assertFalse(tr[0][2])     # first iteration does not stop
        self.assertEqual(tr[-1][0], "offset")
        self.assertTrue(tr[-1][2])     # final iteration reports the stop decision
        for rec in tr:
            self.assertIn(rec[0], ("add", "swap", "offset"))
            if rec[0] == "add":
                self.assertIn(rec[4], ("nearest_core", "nearest_panel"))


if __name__ == "__main__":
    unittest.main()
