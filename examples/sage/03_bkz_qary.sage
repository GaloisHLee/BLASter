"""A small q-ary lattice reduced with one BKZ-3 tour."""

import numpy as np

from blaster import reduce_basis
from blaster.size_reduction import is_weakly_lll_reduced


delta = 0.99
q = 31
basis = matrix(ZZ, [
    [1, 0, 0, 4, 7, 9],
    [0, 1, 0, 8, 2, 5],
    [0, 0, 1, 3, 6, 1],
    [0, 0, 0, q, 0, 0],
    [0, 0, 0, 0, q, 0],
    [0, 0, 0, 0, 0, q],
])

reduced, transform, profile = reduce_basis(
    basis,
    return_transform=True,
    lll_size=3,
    delta=delta,
    cores=1,
    beta=3,
    bkz_tours=1,
    bkz_size=3,
    use_seysen=False,
)

assert transform * basis == reduced
assert abs(transform.det()) == 1
assert abs(reduced.det()) == q**3
assert abs(reduced.det()) == abs(basis.det())
assert reduced.row(0).norm().n() <= basis.row(0).norm().n()

columns = np.asarray(reduced.list(), dtype=np.int64).reshape(reduced.dimensions()).T
upper = np.linalg.qr(columns, mode="r")
assert is_weakly_lll_reduced(upper, delta)

print(f"q-ary lattice: dimension={basis.nrows()}, q={q}")
print("\nBKZ-reduced basis:")
print(reduced)
print(f"\nFirst-vector norm: {basis.row(0).norm().n():.6f} -> "
      f"{reduced.row(0).norm().n():.6f}")
print(profile)
print("BKZ_QARY_OK")
