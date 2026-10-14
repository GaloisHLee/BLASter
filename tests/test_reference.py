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
    """Acceptance #5: the exact reference matches validated FastInt64 decisions.

    Exact == binary64 only when binary64 never rounds a half-integer GSO
    coefficient across a tie. The spec's "Known limitations" records that
    binary64 CAN flip such ties, so bit-equality is asserted only on curated
    inputs with no exact-half ``mu``; the fuzz test instead asserts the exact
    invariant — the reference output always verifies independently.
    """

    def test_deterministic_cases_match(self):
        for name, A in CASES.items():
            with self.subTest(name=name):
                n = A.shape[1]
                U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)

    def test_fuzz_output_always_verifies(self):
        """Random inputs: the exact reference must always verify independently.

        We do NOT assert bit-equality with binary64 ``reduce`` here: near a
        half-integer GSO coefficient binary64 can flip the tie (spec "Known
        limitations"), landing on a different-but-also-valid reduced basis.
        Seed 999 / [-8, 8] deliberately produces such ties so the divergence
        branch is exercised. On any divergence both outputs must still verify
        exactly — i.e. the difference is a binary64 tie-flip, not a reference
        bug.
        """
        rs = np.random.RandomState(999)
        diverged = 0
        for n in range(2, 11):
            for trial in range(60):
                A = rs.randint(-8, 8, size=(n, n)).astype(np.int64)
                try:
                    U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                except Exception:
                    continue  # rank-deficient or non-square reduce rejected
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                rep = verify_reduction(A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99)
                self.assertTrue(rep.verified)
                if not np.array_equal(B_ref, B_fast):
                    diverged += 1
                    rep_fast = verify_reduction(A, B_fast, U_fast, delta=0.99)
                    self.assertTrue(rep_fast.verified)
        # This seed/range contains half-integer ties; assert the divergence
        # path was exercised rather than silently skipped.
        self.assertGreater(diverged, 0)

    def test_half_integer_tie_diverges_from_binary64(self):
        """Pin a concrete binary64 half-integer tie the spec records as a limit.

        ``mu = -3/2`` cannot be stored exactly in binary64; FastInt64's Givens
        path reads ``-1.4999999999999998`` and rounds toward zero, while the
        exact reference rounds the tie away from zero. Both results are valid
        LLL-reduced bases of the same lattice; the exact reference is the
        authoritative one.
        """
        A = np.array([[3, -4], [0, -1]], dtype=np.int64)
        cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
        B_ref = np.array(cols_ref, dtype=np.int64).T
        rep = verify_reduction(A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99)
        self.assertTrue(rep.verified)
        np.testing.assert_array_equal(B_ref, np.array([[-1, 1], [-1, -2]], dtype=np.int64))
        U_fast, B_fast, _ = reduce(A, lll_size=2, delta=0.99, cores=1)
        # binary64 flips the -3/2 tie: FastInt64 lands on a different valid basis.
        self.assertFalse(np.array_equal(B_ref, B_fast))


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


class TestInputValidation(unittest.TestCase):
    """M-2/M-3: the exact reducer rejects invalid input with a clean error."""

    def test_rank_deficient_raises_clean_error(self):
        B0 = [[1, 2], [2, 4]]  # columns (1,2),(2,4) are dependent
        with self.assertRaises(ValueError) as cm:
            lll_reduce(B0, delta=0.99)
        self.assertIn("linearly independent", str(cm.exception))

    def test_delta_out_of_range_raises(self):
        for bad in (0.25, 1.0, 0.1, 2.0):
            with self.subTest(delta=bad):
                with self.assertRaises(ValueError):
                    lll_reduce(CASES["ex3"], delta=bad)

    def test_delta_bool_rejected(self):
        with self.assertRaises(ValueError):
            lll_reduce(CASES["ex3"], delta=True)

    def test_delta_in_range_accepted(self):
        for ok in (0.5, 0.99):
            with self.subTest(delta=ok):
                cols, U, tr = lll_reduce(CASES["ex3"], delta=ok)
                self.assertGreater(len(tr), 0)


class TestRowConvention(unittest.TestCase):
    """M-4: row-oriented entry to the exact reference reducer."""

    def test_lll_reduce_rows_matches_column(self):
        from blaster.reference import lll_reduce_rows
        A0 = CASES["ex3"]  # rows are the basis vectors
        rows, T, tr = lll_reduce_rows(A0, delta=0.99)
        # Row convention: T @ A0 == rows.
        np.testing.assert_array_equal(
            np.array(T, dtype=object) @ np.array(A0, dtype=object),
            np.array(rows, dtype=object),
        )
        # Its transpose is the column-convention reduction of A0.T, and it verifies.
        rep = verify_reduction(A0.T, np.array(rows, dtype=object).T,
                               np.array(T, dtype=object).T, delta=0.99)
        self.assertTrue(rep.verified)


if __name__ == "__main__":
    unittest.main()
