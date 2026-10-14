"""Reference reducer: exact BLASter transcription for ordinary LLL."""
import unittest

import numpy as np

from blaster import reduce, verify_reduction
from blaster.reference import lll_reduce, _progressive_betas, _add_col, _matmul


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


class TestSegmentedLllSize(unittest.TestCase):
    """A-1: ``lll_reduce`` gains a segment width ``lll_size`` matching FastInt64.

    When ``lll_size < n`` the columns partition into disjoint segments and each
    segment is reduced locally in a deterministic sequential order. The exact
    reference is a sequential transcription of FastInt64's parallel
    ``block_lll``; parallelism is a performance property only, out of scope.
    """

    def test_segmented_matches_fastint64(self):
        for name, A in CASES.items():
            n = A.shape[1]
            for lll_size in (2, 3):
                if lll_size >= n:
                    continue
                with self.subTest(name=name, lll_size=lll_size):
                    U_fast, B_fast, _ = reduce(A, lll_size=lll_size, delta=0.99, cores=1)
                    cols_ref, U_ref, _ = lll_reduce(A, delta=0.99, lll_size=lll_size)
                    B_ref = np.array(cols_ref, dtype=np.int64).T
                    np.testing.assert_array_equal(B_ref, B_fast)
                    np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)
                    rep = verify_reduction(A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99)
                    self.assertTrue(rep.verified)

    def test_default_is_single_segment(self):
        A = CASES["rand5"]
        n = A.shape[1]
        cols_full, U_full, tr_full = lll_reduce(A, delta=0.99)
        cols_n, U_n, tr_n = lll_reduce(A, delta=0.99, lll_size=n)
        self.assertEqual(cols_full, cols_n)
        self.assertEqual(U_full, U_n)
        self.assertEqual(tr_full, tr_n)

    def test_lll_size_above_rank_equals_single_segment(self):
        A = CASES["rand5"]
        n = A.shape[1]
        cols_n, U_n, tr_n = lll_reduce(A, delta=0.99, lll_size=n)
        cols_big, U_big, tr_big = lll_reduce(A, delta=0.99, lll_size=n + 10)
        self.assertEqual(cols_n, cols_big)
        self.assertEqual(U_n, U_big)
        self.assertEqual(tr_n, tr_big)

    def test_lll_size_validation(self):
        for bad in (0, 1, -3):
            with self.subTest(lll_size=bad):
                with self.assertRaises(ValueError):
                    lll_reduce(CASES["ex3"], delta=0.99, lll_size=bad)

    def test_reduce_basis_reference_accepts_lll_size(self):
        from blaster.api import reduce_basis
        A = CASES["rand4"]
        cols, U, _ = lll_reduce(A, delta=0.99, lll_size=2)
        red, transform, transcript = reduce_basis(
            A, orientation="columns", engine="reference",
            delta=0.99, lll_size=2, return_transform=True,
        )
        np.testing.assert_array_equal(
            np.array(red, dtype=object), np.array(cols, dtype=object).T
        )
        np.testing.assert_array_equal(
            np.array(transform, dtype=object), np.array(U, dtype=object)
        )


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
            # "offset" and "segment" records carry no column operation
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
            self.assertIn(rec[0], ("add", "swap", "offset", "segment", "deep"))
            if rec[0] == "add":
                self.assertIn(rec[4], ("nearest_core", "nearest_panel"))

    def test_offset_toggle_alternates_half_then_zero(self):
        """A-2: the offset toggles ``block//2 -> 0 -> block//2`` every iteration.

        A direct transcript assertion of the exact sequence, not just that a
        stop happened: every iteration but the closing record alternates
        ``block//2, 0, block//2, 0, ...``, and the closing record repeats the
        final iteration's offset with ``stopped=True``.
        """
        for name, A in CASES.items():
            n = A.shape[1]
            for lll_size in (2, 3, n):
                if lll_size >= n and lll_size != n:
                    continue
                with self.subTest(name=name, lll_size=lll_size):
                    block = min(lll_size, n)
                    cols_ref, U_ref, tr = lll_reduce(A, delta=0.99, lll_size=lll_size)
                    offs = [rec for rec in tr if rec[0] == "offset"]
                    self.assertGreater(len(offs), 1)
                    for idx, rec in enumerate(offs[:-1]):
                        self.assertEqual(
                            rec[1], block // 2 if idx % 2 == 0 else 0,
                            f"iteration {idx} offset",
                        )
                        self.assertFalse(rec[2])
                    self.assertEqual(offs[-1][1], offs[-2][1])
                    self.assertTrue(offs[-1][2])

    def test_exact_three_pass_offset_sequence(self):
        """A-2: pin the exact offset records of a three-pass reduction.

        ``ex3b`` (n=3) with ``lll_size=2`` reduces in three passes — offset 1,
        0, then 1 — before the weak-LLL stop. The transcript records
        ``(1,False), (0,False), (1,False), (1,True)`` exactly.
        """
        cols_ref, U_ref, tr = lll_reduce(CASES["ex3b"], delta=0.99, lll_size=2)
        offs = [rec for rec in tr if rec[0] == "offset"]
        self.assertEqual(
            offs,
            [("offset", 1, False), ("offset", 0, False),
             ("offset", 1, False), ("offset", 1, True)],
        )

    def test_segment_commits_partition_each_phase(self):
        """A-3: each phase's segment transforms commit in ascending order.

        Between two ``offset`` records the transcript lists exactly the
        segments of ``[offset, n)`` — each ``("segment", start, width)`` record
        opening that segment's local-LLL operations — in ascending start order,
        with the classical panel size-reduce following all segments. A verifier
        therefore sees the basis change only at segment boundaries.
        """
        A = CASES["rand5"]  # n = 5
        n = A.shape[1]
        lll_size = 2
        block = min(lll_size, n)
        cols_ref, U_ref, tr = lll_reduce(A, delta=0.99, lll_size=lll_size)

        phases, cur = [], []
        for rec in tr:
            if rec[0] == "offset":
                if cur:
                    phases.append(cur)
                    cur = []
                if rec[2]:
                    break  # closing offset record ends the transcript
            else:
                cur.append(rec)

        offset = 0
        for p in phases:
            offset = (block // 2) if offset == 0 else 0
            self.assertEqual(
                [r for r in p if r[0] == "segment"],
                [("segment", i, min(block, n - i)) for i in range(offset, n, block)],
            )
            start = end = None
            panel = False
            for r in p:
                if r[0] == "segment":
                    self.assertFalse(panel)  # panel is the last step of a phase
                    start, end = r[1], r[1] + r[2]
                elif r[0] == "add" and r[4] == "nearest_panel":
                    panel = True
                else:  # nearest_core add or swap, inside the open segment
                    self.assertFalse(panel)
                    self.assertGreaterEqual(r[1], start)
                    self.assertLess(r[2], end)


class TestReferenceSeysenPanel(unittest.TestCase):
    """B-1: reference ``use_seysen=True`` transcribes the recursive Seysen panel.

    The Seysen panel replaces the classical size-reduction panel: two-column
    Seysen leaves plus rounded-inverse cross-panel, in exact ``Fraction``. The
    reference must reproduce FastInt64 ``reduce(..., use_seysen=True)`` exactly
    on a q-ary corpus, and its output must certify under ``verify_reduction``.
    """

    QARY = {
        "q17_3": np.array([[17, 0, 0], [0, 17, 0], [3, 5, 1]], dtype=np.int64),
        "q17_4": np.array(
            [[17, 0, 0, 0], [0, 17, 0, 0], [0, 0, 17, 0], [3, 5, 7, 1]],
            dtype=np.int64,
        ),
        "q13_5": np.array(
            [[13, 0, 0, 0, 0], [0, 13, 0, 0, 0], [0, 0, 13, 0, 0],
             [0, 0, 0, 13, 0], [2, 4, 6, 8, 1]],
            dtype=np.int64,
        ),
        "q31_6": np.array(
            [[31, 0, 0, 0, 0, 0], [0, 31, 0, 0, 0, 0], [0, 0, 31, 0, 0, 0],
             [0, 0, 0, 31, 0, 0], [0, 0, 0, 0, 31, 0], [5, 7, 9, 11, 13, 1]],
            dtype=np.int64,
        ),
    }

    def test_matches_fastint64_seysen(self):
        for name, A in self.QARY.items():
            with self.subTest(name=name):
                n = A.shape[1]
                U_fast, B_fast, _ = reduce(
                    A, lll_size=n, delta=0.99, cores=1, use_seysen=True
                )
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99, use_seysen=True)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(
                    np.array(U_ref, dtype=np.int64), U_fast
                )
                rep = verify_reduction(
                    A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99
                )
                self.assertTrue(rep.verified, name)

    def test_reduce_basis_reference_accepts_use_seysen(self):
        from blaster.api import reduce_basis
        A = self.QARY["q17_4"]
        cols, U, _ = lll_reduce(A, delta=0.99, use_seysen=True)
        red, transform, transcript = reduce_basis(
            A, orientation="columns", engine="reference",
            delta=0.99, use_seysen=True, return_transform=True,
        )
        np.testing.assert_array_equal(
            np.array(red, dtype=object), np.array(cols, dtype=object).T
        )
        np.testing.assert_array_equal(
            np.array(transform, dtype=object), np.array(U, dtype=object)
        )


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


class TestReferenceDeepLll(unittest.TestCase):
    """C-1: reference ``depth`` transcribes BLASter's DeepLLL.

    ``depth`` truthy swaps ``_local_lll`` for ``_deeplll_local`` inside the
    segment/offset/panel loop (the panel and weak-LLL stop are unchanged).
    The reference must reproduce FastInt64 ``reduce(..., depth=...)`` exactly
    on the DeepLLL corpus, and its transcript must expose the deep-insertion
    decisions through a ``("deep", i, k)`` marker.
    """

    # Curated inputs (found by search) on which BLASter's depth predicate
    # performs at least one *genuine* deep insertion — moving b_k more than
    # one position — so the trajectory test pins a real deep selection, not a
    # plain-LLL Lovász cascade.
    DEEP_CASES = {
        "d2_n8": np.array(
            [[14, 15, 7, 9, 14, -15, 5, 0],
             [14, -1, -5, 8, -12, -10, 5, 1],
             [7, 2, 11, 8, -12, 7, -6, 0],
             [-11, 0, -6, -14, 3, -7, -7, -1],
             [10, -4, 11, -15, -13, -5, -3, -9],
             [-9, 11, -6, 2, -5, 9, 2, 15],
             [12, -14, -14, -10, -2, 12, 3, 7],
             [10, -9, 9, 11, -7, -8, -13, 12]], dtype=np.int64,
        ),
        "d3_n7": np.array(
            [[0, 10, 11, 5, 2, -4, 14],
             [7, 4, 4, -10, 6, 10, 4],
             [0, -2, 4, -15, -3, 2, 2],
             [-7, -1, 14, 13, 13, -10, 3],
             [1, 11, 3, 3, 0, -10, 14],
             [13, -3, -8, 4, -8, -10, 5],
             [6, -10, -12, -15, 1, -2, 2]], dtype=np.int64,
        ),
        "d4_n8": np.array(
            [[-12, 12, 2, -5, 2, -3, -14, 6],
             [-1, 12, 8, -13, 13, -11, -13, -2],
             [-2, 14, -11, -4, -13, 15, 6, 9],
             [-15, 9, -2, 10, 2, 13, 4, -12],
             [1, 1, 3, -8, -12, 2, 1, -7],
             [8, 3, -1, 7, -6, -14, -15, 15],
             [-7, -2, -12, -7, 7, 3, 0, -5],
             [8, 8, -4, 12, 7, -12, 7, 7]], dtype=np.int64,
        ),
    }

    def test_deeplll_matches_fastint64_single_segment(self):
        for name, A in CASES.items():
            n = A.shape[1]
            for depth in (1, 2, 3, 4):
                with self.subTest(name=name, depth=depth):
                    U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1, depth=depth)
                    cols_ref, U_ref, _ = lll_reduce(A, delta=0.99, depth=depth)
                    B_ref = np.array(cols_ref, dtype=np.int64).T
                    np.testing.assert_array_equal(B_ref, B_fast)
                    np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)

    def test_deeplll_matches_fastint64_segmented(self):
        for name, A in CASES.items():
            n = A.shape[1]
            for lll_size in (2, 3):
                if lll_size >= n:
                    continue
                with self.subTest(name=name, lll_size=lll_size):
                    U_fast, B_fast, _ = reduce(
                        A, lll_size=lll_size, delta=0.99, cores=1, depth=2
                    )
                    cols_ref, U_ref, _ = lll_reduce(A, delta=0.99, lll_size=lll_size, depth=2)
                    B_ref = np.array(cols_ref, dtype=np.int64).T
                    np.testing.assert_array_equal(B_ref, B_fast)
                    np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)

    def test_curated_deep_insertions_match_and_verify(self):
        for name, A in self.DEEP_CASES.items():
            depth = {"d2_n8": 2, "d3_n7": 3, "d4_n8": 4}[name]
            with self.subTest(name=name):
                n = A.shape[1]
                U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1, depth=depth)
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99, depth=depth)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)
                rep = verify_reduction(A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99)
                self.assertTrue(rep.verified, name)

    def test_depth_zero_equals_plain_lll(self):
        for name, A in CASES.items():
            with self.subTest(name=name):
                cols_plain, U_plain, tr_plain = lll_reduce(A, delta=0.99)
                cols_zero, U_zero, tr_zero = lll_reduce(A, delta=0.99, depth=0)
                self.assertEqual(cols_plain, cols_zero)
                self.assertEqual(U_plain, U_zero)
                self.assertEqual(tr_plain, tr_zero)

    def test_depth_validation(self):
        with self.assertRaisesRegex(ValueError, "depth must be an integer"):
            lll_reduce(CASES["ex3"], delta=0.99, depth=True)
        with self.assertRaisesRegex(ValueError, "depth must be an integer"):
            lll_reduce(CASES["ex3"], delta=0.99, depth=1.5)
        with self.assertRaisesRegex(ValueError, "depth must be at least 0"):
            lll_reduce(CASES["ex3"], delta=0.99, depth=-1)

    def test_reduce_basis_reference_accepts_depth(self):
        from blaster.api import reduce_basis
        A = CASES["rand4"]
        cols, U, _ = lll_reduce(A, delta=0.99, depth=2)
        red, transform, transcript = reduce_basis(
            A, orientation="columns", engine="reference",
            delta=0.99, depth=2, return_transform=True,
        )
        np.testing.assert_array_equal(
            np.array(red, dtype=object), np.array(cols, dtype=object).T
        )
        np.testing.assert_array_equal(
            np.array(transform, dtype=object), np.array(U, dtype=object)
        )

    def test_transcript_pins_deep_insertion(self):
        """The trajectory exposes a genuine multi-position deep insertion."""
        A = self.DEEP_CASES["d3_n7"]
        depth, n = 3, A.shape[1]
        cols_ref, U_ref, tr = lll_reduce(A, delta=0.99, depth=depth)
        deeps = [rec for rec in tr if rec[0] == "deep"]
        self.assertTrue(deeps, "expected at least one deep-insertion record")
        self.assertTrue(
            any(k - i >= 2 for (_, i, k) in deeps),
            "expected a deep insertion moving b_k more than one position",
        )
        for (_, i, k) in deeps:
            self.assertLess(i, k)
            self.assertLess(k, n)
        # the pinned trajectory replays to the FastInt64 DeepLLL basis exactly
        U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1, depth=depth)
        B_ref = np.array(cols_ref, dtype=np.int64).T
        np.testing.assert_array_equal(B_ref, B_fast)
        np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)


class TestExactColumnAdditionAndProduct(unittest.TestCase):
    """F-1: reference ``_add_col`` / ``_matmul`` stay exact beyond int64.

    Column addition ``M[k] += z*M[i]`` and the reference matrix product are
    the arbitrary-precision "final checker" primitives (§282-284): entries and
    products may exceed signed int64 and the result is the exact closed form,
    never a silent int64 wrap. The checked boundary (``as_int64_matrix``)
    rejects out-of-range int64 input before any reduction begins.
    """

    S = 2 ** 40  # z*entry == 2**80, past int64

    def test_add_col_exact_large_product(self):
        M = [[self.S, self.S], [self.S, self.S]]  # col0=col1=[S, S]
        _add_col(M, 0, 1, self.S)                  # col1 += S*col0
        self.assertEqual(M, [[self.S, self.S],
                             [self.S * (self.S + 1), self.S * (self.S + 1)]])

    def test_add_col_negative_coefficient(self):
        M = [[self.S, 1], [self.S, 2]]  # col0=[S, 1], col1=[S, 2]
        _add_col(M, 0, 1, -self.S)      # col1 += -S*col0
        self.assertEqual(M, [[self.S, 1],
                             [self.S - self.S * self.S, 2 - self.S]])

    def test_matmul_exact_large_product(self):
        A = [[self.S, 1], [0, 1]]
        B = [[self.S], [self.S]]
        self.assertEqual(_matmul(A, B), [[self.S * self.S + self.S], [self.S]])

    def test_matmul_dimension_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "matrix dimension mismatch"):
            _matmul([[1, 2]], [[1, 2]])

    def test_checked_int64_boundary_rejects_out_of_range(self):
        from blaster._matrix import as_int64_matrix
        with self.assertRaises(OverflowError):
            as_int64_matrix([[2 ** 63, 0], [0, 1]], name="basis")


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


class TestProgressiveBetaSchedule(unittest.TestCase):
    """D-1: direct transcription of the FastInt64 ``_progressive_betas`` schedule.

    The schedule is a pure integer function of ``(beta, increment)``; ``beta`` is
    the validated block size (>= 1) and ``increment`` is ``bkz_prog``
    (``1 <= increment <= beta``). It is the reference engine's transcription of
    the upstream progressive-beta sequence used by the segmented BKZ path.
    """

    def test_beta_below_40_returns_empty(self):
        # beta < 40 selects DeepLLL-4, not a pruned BKZ tour: no schedule.
        self.assertEqual(_progressive_betas(1, 1), ())
        self.assertEqual(_progressive_betas(39, 1), ())
        self.assertEqual(_progressive_betas(39, 39), ())

    def test_increment_at_least_beta_returns_single(self):
        # increment >= beta collapses the schedule to the single final beta.
        self.assertEqual(_progressive_betas(40, 40), (40,))
        self.assertEqual(_progressive_betas(45, 45), (45,))
        self.assertEqual(_progressive_betas(45, 50), (45,))
        self.assertEqual(_progressive_betas(50, 80), (50,))

    def test_exact_schedule_tuples(self):
        # Hand-computed: start = 40 + ((beta - 40) % increment), step=increment,
        # inclusive of beta.
        self.assertEqual(_progressive_betas(80, 8), (40, 48, 56, 64, 72, 80))
        self.assertEqual(_progressive_betas(79, 5), (44, 49, 54, 59, 64, 69, 74, 79))
        self.assertEqual(_progressive_betas(60, 10), (40, 50, 60))
        self.assertEqual(_progressive_betas(59, 10), (49, 59))
        self.assertEqual(_progressive_betas(43, 3), (40, 43))
        self.assertEqual(_progressive_betas(42, 3), (42,))
        self.assertEqual(_progressive_betas(41, 3), (41,))

    def test_every_schedule_starts_in_first_window_and_ends_at_beta(self):
        for beta in range(40, 81):
            for increment in range(1, beta + 1):
                s = _progressive_betas(beta, increment)
                self.assertGreaterEqual(len(s), 1)
                self.assertEqual(s[-1], beta, (beta, increment))
                self.assertGreaterEqual(s[0], 40, (beta, increment))
                self.assertLess(s[0], 40 + increment, (beta, increment))

    def test_matches_fastint64_source_of_truth(self):
        from blaster.blaster import _progressive_betas as _fast_progressive_betas
        for beta in range(1, 81):
            for increment in range(1, beta + 1):
                self.assertEqual(
                    _progressive_betas(beta, increment),
                    _fast_progressive_betas(beta, increment),
                    (beta, increment),
                )


class TestReductionTreeSchedule(unittest.TestCase):
    """F-4: the balanced reduction-tree scheduler, pinned at its contract.

    ``__reduction_ranges`` (FastInt64 ``size_reduction``) and ``_reduction_ranges``
    (reference ``reference``) build the balanced binary reduction tree:
    ``base_cases`` are adjacent pairs reduced directly, and each triple
    ``(u, v, w)`` marks the split point ``v`` where ``[v:w)`` is reduced w.r.t.
    ``[u:v)``. ``__babai_ranges`` maps each split index ``v`` to its deferred
    propagation span ``(u, w)``.

    The docstring's recursive reference realises the *same contract* with a
    different (unrolled/merged) tree: its ``F(l, r)`` splits ``[l, r)`` at
    ``(l + r) // 2`` and emits a length-2 ``[l, l+1)`` as base case ``l`` plus a
    triple for every longer split, whereas the non-recursive implementation
    unrolls length-2 intervals *into* their parent triple and records fewer
    triples. These tests therefore pin the *contract* (the documented coverage
    guarantee) and the implementation's own golden output — not a literal match
    to the recursive reference.
    """

    # Golden output of the non-recursive implementation (hand-verified, n=1..9).
    GOLDEN_RANGES = {
        1: ([], []),
        2: ([0], []),
        3: ([1], [(0, 1, 3)]),
        4: ([0, 2], [(0, 2, 4)]),
        5: ([0, 3], [(2, 3, 5), (0, 2, 5)]),
        6: ([1, 4], [(3, 4, 6), (0, 1, 3), (0, 3, 6)]),
        7: ([1, 3, 5], [(3, 5, 7), (0, 1, 3), (0, 3, 7)]),
        8: ([0, 2, 4, 6], [(4, 6, 8), (0, 2, 4), (0, 4, 8)]),
        9: ([0, 2, 4, 7], [(6, 7, 9), (4, 6, 9), (0, 2, 4), (0, 4, 9)]),
    }

    GOLDEN_BABAI = {
        1: [False],
        2: [False, False],
        3: [False, (0, 3), False],
        4: [False, False, (0, 4), False],
        5: [False, False, (0, 5), (2, 5), False],
        6: [False, (0, 3), False, (0, 6), (3, 6), False],
        7: [False, (0, 3), False, (0, 7), False, (3, 7), False],
        8: [False, False, (0, 4), False, (0, 8), False, (4, 8), False],
        9: [False, False, (0, 4), False, (0, 9), False, (4, 9), (6, 9), False],
    }

    @staticmethod
    def _fast_ranges(n):
        # `__reduction_ranges` is a name-mangled private; import via getattr so
        # the double-underscore identifier is not mangled inside this class body.
        import importlib
        mod = importlib.import_module("blaster.size_reduction")
        return getattr(mod, "__reduction_ranges")(n)

    @staticmethod
    def _fast_babai(n):
        import importlib
        mod = importlib.import_module("blaster.size_reduction")
        return getattr(mod, "__babai_ranges")(n)

    @staticmethod
    def _ref_ranges(n):
        from blaster.reference import _reduction_ranges
        return _reduction_ranges(n)

    def test_reduction_ranges_golden(self):
        for n, (base, ranges) in self.GOLDEN_RANGES.items():
            with self.subTest(n=n):
                self.assertEqual(self._fast_ranges(n), (base, ranges))
                self.assertEqual(self._ref_ranges(n), (base, ranges))

    def test_babai_ranges_golden(self):
        for n, babai in self.GOLDEN_BABAI.items():
            with self.subTest(n=n):
                self.assertEqual(self._fast_babai(n), babai)

    def test_reference_matches_fastint64(self):
        for n in range(1, 81):
            with self.subTest(n=n):
                self.assertEqual(self._ref_ranges(n), self._fast_ranges(n))

    def test_every_column_pair_is_covered(self):
        # The documented guarantee: for every 0 <= i < j < n, either j == i + 1
        # with i a base case, or some triple (u, v, w) separates i and j
        # (u <= i < v <= j < w). This is what makes the scheduler complete: no
        # pair of columns is ever left unreduced.
        for n in range(1, 61):
            base, ranges = self._fast_ranges(n)
            base = set(base)
            for i in range(n):
                for j in range(i + 1, n):
                    covered = (j == i + 1 and i in base) or any(
                        u <= i < v <= j < w for (u, v, w) in ranges
                    )
                    self.assertTrue(covered, (n, i, j))

    def test_triples_and_base_cases_are_valid_indices(self):
        for n in range(1, 81):
            base, ranges = self._fast_ranges(n)
            for b in base:
                self.assertGreaterEqual(b, 0, (n, b))
                self.assertLess(b, n, (n, b))
            for (u, v, w) in ranges:
                self.assertGreaterEqual(u, 0, (n, (u, v, w)))
                self.assertLess(u, v, (n, (u, v, w)))
                self.assertLess(v, w, (n, (u, v, w)))
                self.assertLessEqual(w, n, (n, (u, v, w)))

    def test_babai_marks_each_split_and_only_splits(self):
        # ``__babai_ranges`` maps a split index ``v`` to its ``(u, w)`` span;
        # every triple's middle owns its span, and every non-middle index stays
        # ``False`` (a plain base case, no deferred propagation).
        for n in range(1, 81):
            _, ranges = self._fast_ranges(n)
            babai = self._fast_babai(n)
            split_to_span = {v: (u, w) for (u, v, w) in ranges}
            for j in range(n):
                self.assertEqual(babai[j], split_to_span.get(j, False), (n, j))


class TestFrontTourSchedule(unittest.TestCase):
    """F-4: the BKZ front/tour scheduler advances by the segment stride and
    only wraps to a new tour on a *strict* overrun (``cur_front + beta > n``).

    An exact end-of-rank fit (``cur_front + beta == n``) must still advance —
    implementation-map row 113 flags the strict ``>`` as easy to misread as ``>=``.
    """

    def setUp(self):
        from blaster.blaster import _advance_front
        self._advance_front = _advance_front

    def test_stride_is_segment_size_minus_beta_plus_one(self):
        # n=10, beta=3, bkz_size=6 -> stride 6 - 3 + 1 = 4.
        self.assertEqual(self._advance_front(0, 3, 10, 6), (4, 0))
        self.assertEqual(self._advance_front(4, 3, 10, 6), (8, 0))

    def test_exact_end_of_rank_fit_advances_does_not_wrap(self):
        # cur_front + beta == n is NOT an overrun: strict '>' is the wrap gate.
        self.assertEqual(self._advance_front(7, 3, 10, 6), (11, 0))
        self.assertEqual(self._advance_front(6, 4, 10, 6), (9, 0))

    def test_strict_overrun_wraps_and_counts_a_tour(self):
        # cur_front + beta > n is the only wrap condition.
        self.assertEqual(self._advance_front(8, 3, 10, 6), (0, 1))
        self.assertEqual(self._advance_front(7, 4, 10, 6), (0, 1))

    def test_unit_stride_when_segment_equals_beta(self):
        # bkz_size == beta collapses the stride to 1.
        self.assertEqual(self._advance_front(0, 3, 10, 3), (1, 0))
        self.assertEqual(self._advance_front(8, 3, 10, 3), (0, 1))

    def test_full_tour_visits_each_front_then_wraps(self):
        fronts = []
        cur_front, tours_done = 0, 0
        while tours_done == 0:
            cur_front, delta = self._advance_front(cur_front, 3, 10, 6)
            tours_done += delta
            if tours_done == 0:
                fronts.append(cur_front)
        self.assertEqual(fronts, [4, 8])
        self.assertEqual(tours_done, 1)


if __name__ == "__main__":
    unittest.main()
