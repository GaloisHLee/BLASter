"""Reference reducer: exact BLASter transcription for ordinary LLL."""
import unittest

import numpy as np

from blaster import reduce, verify_reduction
from blaster.reference import lll_reduce, _progressive_betas


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


if __name__ == "__main__":
    unittest.main()
