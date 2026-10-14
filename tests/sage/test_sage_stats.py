"""Sage-only tests: ``stats.get_profile`` accepts Sage integer matrices."""
import unittest

from sage.all import matrix, ZZ

from blaster.stats import get_profile


class TestGetProfileSageMatrix(unittest.TestCase):
    def test_get_profile_accepts_arbitrary_precision_sage_matrix(self):
        # Entries beyond int64 force the object-dtype exact path, which must
        # accept a Sage matrix rather than assuming a NumPy array.
        big = 1 << 2000
        basis = matrix(ZZ, [[big, 0], [0, 1]])
        self.assertEqual(get_profile(basis), [2000.0, 0.0])

    def test_get_profile_sage_matches_numpy_object_array(self):
        big = 1 << 70
        basis = matrix(ZZ, [[big, 0, 0], [0, big, 0], [big // 2, big // 3, 1]])
        import numpy as np
        from blaster.stats import _object_profile

        expected = _object_profile(
            np.asarray([[big, 0, 0], [0, big, 0], [big // 2, big // 3, 1]], dtype=object)
        )
        self.assertEqual(get_profile(basis), expected)

    def test_get_profile_sage_matrix_is_upper(self):
        # The is_upper shortcut reads ``upper.diagonal()`` and logs each entry
        # directly; a Sage integer must not be coerced through float64, which
        # overflows for entries beyond that range.
        big = 1 << 2000
        upper = matrix(ZZ, [[big, big], [0, 1]])
        self.assertEqual(get_profile(upper, is_upper=True), [2000.0, 0.0])


if __name__ == "__main__":
    unittest.main()
