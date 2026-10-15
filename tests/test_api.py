import unittest

import numpy as np

from blaster import reduce, reduce_basis, verify_reduction
from blaster.size_reduction import is_weakly_lll_reduced


class ReductionApiTests(unittest.TestCase):
    def setUp(self):
        # The low-level API uses column basis vectors.
        self.columns = np.array(
            [
                [10, 1, 0],
                [0, 10, 1],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )

    def test_reduce_columns_preserves_lattice_and_input(self):
        original = self.columns.copy()

        transform, reduced, profile = reduce(
            self.columns, lll_size=3, cores=1, use_seysen=False
        )

        np.testing.assert_array_equal(self.columns, original)
        np.testing.assert_array_equal(original @ transform, reduced)
        self.assertEqual(abs(round(np.linalg.det(transform))), 1)
        self.assertTrue(
            is_weakly_lll_reduced(np.linalg.qr(reduced, mode="r"), delta=0.99)
        )
        self.assertGreaterEqual(profile.num_iterations, 1)

    def test_reduce_basis_defaults_to_row_vectors(self):
        rows = self.columns.T.copy()

        reduced, transform, _ = reduce_basis(
            rows,
            return_transform=True,
            lll_size=3,
            cores=1,
            use_seysen=False,
        )

        np.testing.assert_array_equal(transform @ rows, reduced)
        self.assertEqual(abs(round(np.linalg.det(transform))), 1)

    def test_reduce_basis_supports_column_orientation(self):
        reduced, transform, _ = reduce_basis(
            self.columns,
            orientation="columns",
            return_transform=True,
            lll_size=3,
            cores=1,
            use_seysen=False,
        )

        np.testing.assert_array_equal(self.columns @ transform, reduced)

    def test_reduce_basis_supports_rectangular_row_basis(self):
        rows = np.array([[7, 1, 0], [2, 5, 1]], dtype=np.int64)

        reduced, transform, _ = reduce_basis(
            rows,
            return_transform=True,
            lll_size=2,
            cores=1,
            use_seysen=False,
        )

        self.assertEqual(reduced.shape, rows.shape)
        np.testing.assert_array_equal(transform @ rows, reduced)

    def test_small_bkz_run_preserves_lattice(self):
        transform, reduced, _ = reduce(
            self.columns,
            lll_size=3,
            cores=1,
            beta=3,
            bkz_tours=1,
            bkz_size=3,
            use_seysen=False,
        )

        np.testing.assert_array_equal(self.columns @ transform, reduced)
        self.assertEqual(abs(round(np.linalg.det(transform))), 1)

    def test_rank_one_basis_is_supported(self):
        basis = np.array([[5]], dtype=np.int64)

        transform, reduced, profile = reduce(basis, cores=1)

        np.testing.assert_array_equal(transform, np.array([[1]], dtype=np.int64))
        np.testing.assert_array_equal(reduced, basis)
        self.assertEqual(profile.num_iterations, 1)

    def test_rejects_non_integer_and_out_of_range_entries(self):
        with self.assertRaisesRegex(TypeError, "integer"):
            reduce_basis(np.eye(2, dtype=np.float64))

        with self.assertRaisesRegex(TypeError, "integer"):
            reduce_basis([[True, 0], [0, True]])

        with self.assertRaisesRegex(OverflowError, "int64"):
            reduce_basis([[2**63, 0], [0, 1]])

    def test_rejects_invalid_basis_shapes(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            reduce_basis([1, 2, 3])

        with self.assertRaisesRegex(ValueError, "ambient dimension"):
            reduce_basis(np.ones((3, 2), dtype=np.int64), orientation="rows")

        with self.assertRaisesRegex(ValueError, "linearly independent"):
            reduce_basis(np.array([[1, 2], [2, 4]], dtype=np.int64))

        with self.assertRaisesRegex(ValueError, "orientation"):
            reduce_basis(np.eye(2, dtype=np.int64), orientation="diagonal")

    def test_rank_check_is_exact_for_wide_dynamic_range(self):
        from blaster.blaster import _validated_parameters

        # Full rank (det = 1), but a float64 SVD reads it as rank 1 because the
        # O(1) column is swamped by the O(2**50) column's tolerance.
        wide = np.array([[1, 0], [2**50, 1]], dtype=np.int64)
        validated = _validated_parameters(wide, 2, 0.99, 1, 0, None, 8, None, None)
        np.testing.assert_array_equal(validated[0], wide)

        # A genuinely rank-deficient basis is still rejected.
        dependent = np.array([[1, 2], [2, 4]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "linearly independent"):
            _validated_parameters(dependent, 2, 0.99, 1, 0, None, 8, None, None)

    def test_rejects_invalid_reduction_parameters(self):
        invalid_cases = [
            ({"delta": 0.25}, "delta"),
            ({"lll_size": 1}, "lll_size"),
            ({"cores": 0}, "cores"),
            ({"depth": -1}, "depth"),
            ({"depth": 2, "beta": 3}, "depth.*beta"),
            ({"beta": 81}, "beta"),
            ({"beta": 3, "bkz_tours": 0}, "bkz_tours"),
            ({"beta": 3, "bkz_size": 2}, "bkz_size"),
            ({"beta": 3, "bkz_prog": 0}, "bkz_prog"),
        ]

        for parameters, message in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, message):
                    call_parameters = {"cores": 1, **parameters}
                    reduce(self.columns, **call_parameters)


class MultiCoreSwitchTests(unittest.TestCase):
    """M-2: ``cores`` resolves to ``"auto"`` (max) by default, or an explicit int."""

    def setUp(self):
        self.columns = np.array(
            [[10, 1, 0], [0, 10, 1], [0, 0, 1]], dtype=np.int64
        )

    def test_resolve_cores_auto_detects_cpu_count(self):
        import os as _os
        from blaster.blaster import _resolve_cores

        detected = _os.cpu_count()
        expected = detected if (detected is not None and detected >= 1) else 1
        for spelling in ("auto", "Auto", "AUTO", " auto "):
            with self.subTest(spelling=spelling):
                self.assertEqual(_resolve_cores(spelling), expected)

    def test_resolve_cores_explicit_integer(self):
        from blaster.blaster import _resolve_cores

        self.assertEqual(_resolve_cores(1), 1)
        self.assertEqual(_resolve_cores(4), 4)

    def test_resolve_cores_rejects_invalid_values(self):
        from blaster.blaster import _resolve_cores

        for bad in (0, True, "bogus", "4", 2.5, None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "cores"):
                    _resolve_cores(bad)

    def test_resolve_cores_falls_back_to_one_when_cpu_count_is_none(self):
        from unittest import mock
        import blaster.blaster as blaster_module

        with mock.patch.object(blaster_module.os, "cpu_count", return_value=None):
            self.assertEqual(blaster_module._resolve_cores("auto"), 1)

    def test_reduce_cores_auto_matches_single_threaded(self):
        t1, r1, _ = reduce(self.columns, lll_size=3, cores=1, use_seysen=False)
        ta, ra, _ = reduce(self.columns, lll_size=3, cores="auto", use_seysen=False)

        np.testing.assert_array_equal(r1, ra)
        np.testing.assert_array_equal(t1, ta)

    def test_reduce_defaults_to_auto_cores(self):
        import os as _os
        from unittest import mock
        import blaster.blaster as blaster_module

        detected = _os.cpu_count()
        expected = detected if (detected is not None and detected >= 1) else 1
        with mock.patch.object(blaster_module, "set_num_cores") as set_cores:
            reduce(self.columns, lll_size=3, use_seysen=False)
            set_cores.assert_called_once_with(expected)

    def test_reduce_basis_forwards_cores_auto(self):
        rows = self.columns.T.copy()
        reduced_1 = reduce_basis(rows, lll_size=3, cores=1, use_seysen=False)
        reduced_auto = reduce_basis(rows, lll_size=3, cores="auto", use_seysen=False)

        np.testing.assert_array_equal(reduced_1, reduced_auto)


class CompletionStatusTests(unittest.TestCase):
    """F-2: ``reduce`` surfaces whether reduction finished normally or was
    interrupted, so a caller can tell a fully reduced basis from a partial one.

    The interruption is injected by raising ``KeyboardInterrupt`` from the
    reduction entry point — the exact signal the engine is documented to catch
    and turn into a partial-state return rather than propagating.
    """

    def setUp(self):
        self.columns = np.array(
            [[10, 1, 0], [0, 10, 1], [0, 0, 1]], dtype=np.int64
        )

    def test_normal_completion_marks_profile_not_interrupted(self):
        _, _, profile = reduce(self.columns, lll_size=3, cores=1, use_seysen=False)
        self.assertFalse(profile.interrupted)

    def test_interrupted_run_marks_profile_interrupted_and_returns_partial_state(self):
        import blaster.blaster as blaster_module
        from unittest import mock

        with mock.patch.object(
            blaster_module, "lll_reduce", side_effect=KeyboardInterrupt
        ):
            transform, reduced, profile = reduce(
                self.columns, lll_size=3, cores=1, use_seysen=False
            )

        self.assertTrue(profile.interrupted)
        # The partial state still comes back in the same (transform, basis,
        # profile) tuple shape; no exception escapes the engine.
        self.assertEqual(transform.shape, (3, 3))
        self.assertEqual(reduced.shape, self.columns.shape)


class ReferenceEngineTests(unittest.TestCase):
    """engine="reference": exact arbitrary-precision LLL via the high-level API."""

    def test_reference_engine_accepts_arbitrary_precision_entries(self):
        big = 1 << 70  # exceeds int64; the fast engine would raise OverflowError
        rows = [[big, 0, 0], [0, big, 0], [big // 2, big // 3, 1]]
        reduced, transform, transcript = reduce_basis(
            rows, engine="reference", return_transform=True
        )
        np.testing.assert_array_equal(
            np.array(transform, dtype=object) @ np.array(rows, dtype=object),
            np.array(reduced, dtype=object),
        )
        rep = verify_reduction(
            np.array(rows, dtype=object).T,
            np.array(reduced, dtype=object).T,
            np.array(transform, dtype=object).T,
            delta=0.99,
        )
        self.assertTrue(rep.verified)
        self.assertTrue(transcript)  # non-empty decision log

    def test_reference_engine_supports_column_orientation(self):
        basis = np.array([[10, 0, 0], [1, 10, 0], [0, 1, 1]], dtype=np.int64)
        reduced, transform, _ = reduce_basis(
            basis, orientation="columns", engine="reference", return_transform=True
        )
        np.testing.assert_array_equal(
            np.array(basis, dtype=object) @ np.array(transform, dtype=object),
            np.array(reduced, dtype=object),
        )

    def test_reference_engine_rejects_unsupported_parameters(self):
        with self.assertRaisesRegex(ValueError, "delta"):
            reduce_basis([[1, 0], [0, 1]], engine="reference", cores=4)

    def test_reference_engine_accepts_gso_selector(self):
        rows = [[10, 0, 0], [1, 10, 0], [0, 1, 1]]
        red_def, T_def, tr_def = reduce_basis(rows, engine="reference", return_transform=True)
        red_exp, T_exp, tr_exp = reduce_basis(
            rows, engine="reference", gso="rational", return_transform=True
        )
        np.testing.assert_array_equal(red_def, red_exp)
        np.testing.assert_array_equal(T_def, T_exp)
        self.assertEqual(tr_def, tr_exp)

    def test_reference_engine_rejects_invalid_gso(self):
        with self.assertRaisesRegex(ValueError, "gso"):
            reduce_basis([[1, 0], [0, 1]], engine="reference", gso="numpy")

    def test_reference_engine_rejects_invalid_engine(self):
        with self.assertRaisesRegex(ValueError, "engine"):
            reduce_basis([[1, 0], [0, 1]], engine="quantum")


class ReferenceBKZApiTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array(
            [[2, 3, 5, 7], [0, 4, 6, 8], [0, 0, 7, 9], [0, 0, 0, 11]], dtype=object
        )

    def test_reference_bkz_returns_reduced(self):
        reduced, transform, _ = reduce_basis(
            self.rows, engine="reference", beta=2, return_transform=True
        )
        from blaster.verify import verify_reduction
        rep = verify_reduction(
            self.rows.T.tolist(),
            np.array(reduced, dtype=object).T.tolist(),
            np.array(transform, dtype=object).T.tolist(),
            mode="bkz",
        )
        self.assertTrue(rep.structural_ok)

    def test_reference_bkz_return_transform(self):
        reduced, transform, transcript = reduce_basis(
            self.rows, engine="reference", beta=2, return_transform=True
        )
        self.assertIsInstance(transcript, list)

    def test_reference_bkz_rejects_depth(self):
        with self.assertRaisesRegex(ValueError, "depth"):
            reduce_basis(self.rows, engine="reference", beta=2, depth=2)

    def test_reference_still_rejects_unknown_params(self):
        with self.assertRaisesRegex(ValueError, "cores"):
            reduce_basis(self.rows, engine="reference", cores=1)


class PublicExportsTests(unittest.TestCase):
    """Exact reference reducer and Sage accelerator are public on ``blaster``."""

    def test_lll_reduce_rows_is_public(self):
        from blaster import lll_reduce_rows
        from blaster.reference import lll_reduce_rows as _ref
        self.assertIs(lll_reduce_rows, _ref)
        rows = [[10, 0, 0], [1, 10, 0], [0, 1, 1]]
        reduced, T, transcript = lll_reduce_rows(rows, delta=0.99)
        np.testing.assert_array_equal(
            np.array(T, dtype=object) @ np.array(rows, dtype=object),
            np.array(reduced, dtype=object),
        )
        self.assertTrue(transcript)

    def test_sage_gso_is_public(self):
        from blaster import sage_gso
        from blaster.verify_sage import sage_gso as _ref
        self.assertIs(sage_gso, _ref)
        cols = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        mu, r = sage_gso(cols)
        self.assertEqual(len(mu), 3)
        self.assertEqual(len(r), 3)


if __name__ == "__main__":
    unittest.main()
