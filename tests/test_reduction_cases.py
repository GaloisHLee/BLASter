import unittest

import numpy as np

from blaster import reduce
from blaster.size_reduction import is_weakly_lll_reduced


def _assert_valid_reduction(test_case, basis, parameters):
    transform, reduced, _ = reduce(basis, **parameters)
    np.testing.assert_array_equal(basis @ transform, reduced)
    test_case.assertEqual(abs(round(np.linalg.det(transform))), 1)
    test_case.assertTrue(
        is_weakly_lll_reduced(np.linalg.qr(reduced, mode="r"), parameters["delta"])
    )


class RepresentativeLatticeTests(unittest.TestCase):
    def test_deterministic_small_random_lattices(self):
        rng = np.random.default_rng(20260808)
        for dimension in range(2, 9):
            basis = np.identity(dimension, dtype=np.int64)
            for _ in range(4 * dimension):
                source, target = rng.choice(dimension, size=2, replace=False)
                basis[:, target] += int(rng.choice([-3, -2, 2, 3])) * basis[:, source]

            with self.subTest(dimension=dimension):
                _assert_valid_reduction(
                    self,
                    basis,
                    {
                        "lll_size": min(4, dimension),
                        "delta": 0.99,
                        "cores": 1,
                        "use_seysen": False,
                    },
                )

    def test_qary_lattice_with_seysen_and_parallel_segments(self):
        q = 31
        row_basis = np.array(
            [
                [1, 0, 0, 4, 7, 9],
                [0, 1, 0, 8, 2, 5],
                [0, 0, 1, 3, 6, 1],
                [0, 0, 0, q, 0, 0],
                [0, 0, 0, 0, q, 0],
                [0, 0, 0, 0, 0, q],
            ],
            dtype=np.int64,
        )
        basis = np.ascontiguousarray(row_basis.T)

        _assert_valid_reduction(
            self,
            basis,
            {
                "lll_size": 3,
                "delta": 0.99,
                "cores": 2,
                "use_seysen": True,
            },
        )

    def test_deep_lll_preserves_lattice(self):
        basis = np.array(
            [
                [17, 3, 2, 1],
                [0, 13, 4, 2],
                [0, 0, 7, 3],
                [0, 0, 0, 2],
            ],
            dtype=np.int64,
        )

        _assert_valid_reduction(
            self,
            basis,
            {
                "lll_size": 4,
                "delta": 0.99,
                "cores": 1,
                "depth": 2,
                "use_seysen": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
