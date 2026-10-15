import math
import unittest
from fractions import Fraction

from blaster._bkz import (
    PRUNING_PARAMS,
    _gh_squared,
    _safe_gh_squared,
    enumeration,
    get_pruning_coefficients,
)
from blaster._exact import column, delta_to_fraction, to_int_matrix, transpose_cols
from blaster.reference import _bkz_segment, _deeplll_local, bkz_reduce
from blaster.verify import verify_reduction
from blaster.verify_sage import resolve_gso


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


class TestEnumeration(unittest.TestCase):
    def test_single_dim_finds_shortest_nonzero(self):
        # 1-dim block, r=[4] (= ||b*_0||^2, so ||b*_0||=2), radius=5 ->
        # shortest nonzero coefficient is +-1 (norm 1^2 * 4 = 4)
        mu = [[Fraction(0)]]
        r = [Fraction(4)]
        pr = [Fraction(1)]
        sol, norm = enumeration(mu, r, pr, Fraction(5))
        self.assertEqual(norm, Fraction(4))
        self.assertEqual(sol, [1])

    def test_single_dim_no_solution_is_zero_norm(self):
        # radius already <= shortest nonzero norm -> returns (sol, 0)
        mu = [[Fraction(0)]]
        r = [Fraction(4)]
        pr = [Fraction(1)]
        _, norm = enumeration(mu, r, pr, Fraction(4))
        self.assertEqual(norm, Fraction(0))

    def test_two_dim_unpruned_finds_combination(self):
        # r=[1, 1/4], mu[0][1]=3/4, no pruning. norm^2(x0,x1) =
        # (x0 + 3/4 x1)^2 + x1^2/4; minimum over Z^2\{0} is (-1,1) -> 5/16.
        mu = [[Fraction(0), Fraction(3, 4)], [Fraction(0), Fraction(0)]]
        r = [Fraction(1), Fraction(1, 4)]
        pr = [Fraction(1), Fraction(1)]
        sol, norm = enumeration(mu, r, pr, Fraction(1))
        self.assertEqual(norm, Fraction(5, 16))
        self.assertEqual(sol, [-1, 1])


def _deeplll4_reference(B0, delta=0.99):
    B0_rows = to_int_matrix(B0, name="B0")
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []
    _deeplll_local(cols, Ucols, 0, n, d, 4, tr, resolve_gso(None))
    return cols, transpose_cols(Ucols), tr


class TestBKZReduce(unittest.TestCase):
    def test_bkz2_is_deeplll4(self):
        B0 = [[2, 3, 5], [0, 4, 6], [0, 0, 7]]
        cols_a, U_a, tr_a = bkz_reduce(B0, beta=2)
        cols_b, U_b, tr_b = _deeplll4_reference(B0)
        self.assertEqual(cols_a, cols_b)
        self.assertEqual(U_a, U_b)
        self.assertEqual(tr_a, tr_b)

    def test_bkz_tour_verifies(self):
        B0 = [[2, 3, 5, 7, 11], [0, 4, 6, 8, 10], [0, 0, 7, 9, 12],
              [0, 0, 0, 11, 13], [0, 0, 0, 0, 17]]
        cols, U_rows, tr = bkz_reduce(B0, beta=3)
        rep = verify_reduction(B0, transpose_cols(cols), U_rows, mode="bkz")
        self.assertTrue(rep.verified)

    def test_hkz_tour_verifies(self):
        B0 = [[2, 3, 5], [0, 4, 6], [0, 0, 7]]
        cols, U_rows, tr = bkz_reduce(B0, beta=4)
        rep = verify_reduction(B0, transpose_cols(cols), U_rows, mode="bkz")
        self.assertTrue(rep.verified)

    def test_bad_beta_rejected(self):
        B0 = [[2, 3], [0, 4]]
        for bad in (1, 81, 3.5, True):
            with self.assertRaises(ValueError):
                bkz_reduce(B0, beta=bad)


class TestBKZSegment(unittest.TestCase):
    def _columns(self, rows):
        n = len(rows[0])
        d = delta_to_fraction(0.99)
        cols = [column(to_int_matrix(rows), j) for j in range(n)]
        Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
        return cols, Ucols, [], d

    def test_bkz_segment_hkz_tail_verifies(self):
        # beta > segment width forces an HKZ sweep inside the segment.
        B0 = [[2, 3, 5, 7, 11, 13], [0, 4, 6, 8, 10, 12], [0, 0, 7, 9, 12, 15],
              [0, 0, 0, 11, 13, 17], [0, 0, 0, 0, 17, 19], [0, 0, 0, 0, 0, 23]]
        cols, Ucols, tr, d = self._columns(B0)
        _bkz_segment(cols, Ucols, 0, 6, 7, d, tr, resolve_gso(None))  # full segment [0,6), beta 7 > 6 -> HKZ
        rep = verify_reduction(B0, transpose_cols(cols), transpose_cols(Ucols), mode="bkz")
        self.assertTrue(rep.verified)

    def test_bkz_size_below_beta_rejected(self):
        from blaster.reference import _bkz_size_param
        for bad in (3, 1, True, 2.5):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "bkz_size"):
                    _bkz_size_param(bad, 4, 6)

    def test_bkz_size_capped_at_rank(self):
        from blaster.reference import _bkz_size_param
        self.assertEqual(_bkz_size_param(10, 4, 6), 6)


class TestVerifyBKZ(unittest.TestCase):
    def test_bkz_mode_accepted_and_verifies(self):
        B0 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        B = B0
        U = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rep = verify_reduction(B0, B, U, mode="bkz")
        self.assertTrue(rep.verified)
        self.assertEqual(rep.mode, "bkz")
        self.assertTrue(rep.size_reduced_ok)
        self.assertTrue(rep.deep_ok)

    def test_bkz_mode_rejects_not_deep(self):
        # size-reduced but NOT deep-LLL-4: b_1 is far shorter than b_0's norm.
        B0 = [[10, 0], [0, 1]]
        B = B0
        U = [[1, 0], [0, 1]]
        rep = verify_reduction(B0, B, U, mode="bkz")
        self.assertFalse(rep.verified)
        self.assertTrue(rep.structural_ok)
        self.assertTrue(rep.size_reduced_ok)
        self.assertFalse(rep.deep_ok)

    def test_bkz_mode_ignores_depth_kwarg(self):
        # depth=1 must NOT override the fixed depth-4 gating for bkz.
        B0 = [[10, 0], [0, 1]]
        B = B0
        U = [[1, 0], [0, 1]]
        rep = verify_reduction(B0, B, U, mode="bkz", depth=1)
        self.assertFalse(rep.verified)
        self.assertFalse(rep.deep_ok)


if __name__ == "__main__":
    unittest.main()
