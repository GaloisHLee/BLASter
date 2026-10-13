"""BLASter's column-vector convention through the high-level Sage API."""

import numpy as np

from blaster import reduce_basis
from blaster.size_reduction import is_weakly_lll_reduced


delta = 0.99
# Two basis vectors in the three-dimensional ambient space, stored as columns.
basis = matrix(ZZ, [
    [7, 2],
    [1, 5],
    [0, 1],
])

reduced, transform, profile = reduce_basis(
    basis,
    orientation="columns",
    return_transform=True,
    lll_size=2,
    delta=delta,
    cores=1,
    use_seysen=True,
)

assert basis * transform == reduced
assert abs(transform.det()) == 1
assert reduced.column(0).norm().n() <= basis.column(0).norm().n()

columns = np.asarray(reduced.list(), dtype=np.int64).reshape(reduced.dimensions())
upper = np.linalg.qr(columns, mode="r")
assert is_weakly_lll_reduced(upper, delta)

print("Original column basis:")
print(basis)
print("\nReduced column basis:")
print(reduced)
print(f"\nFirst-vector norm: {basis.column(0).norm().n():.6f} -> "
      f"{reduced.column(0).norm().n():.6f}")
print(profile)
print("COLUMN_BASIS_OK")
