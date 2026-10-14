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

    def test_reference_engine_rejects_invalid_engine(self):
        with self.assertRaisesRegex(ValueError, "engine"):
            reduce_basis([[1, 0], [0, 1]], engine="quantum")


if __name__ == "__main__":
    unittest.main()
