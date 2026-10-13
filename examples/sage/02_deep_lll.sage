"""DeepLLL-2 with Seysen reduction on a four-dimensional lattice."""

import numpy as np

from blaster import reduce_basis
from blaster.size_reduction import is_weakly_lll_reduced


delta = 0.99
basis = matrix(ZZ, [
    [17, 0, 0, 0],
    [3, 13, 0, 0],
    [2, 4, 7, 0],
    [1, 2, 3, 2],
])

reduced, transform, profile = reduce_basis(
    basis,
    return_transform=True,
    lll_size=4,
    delta=delta,
    cores=1,
    depth=2,
    use_seysen=True,
)

assert transform * basis == reduced
assert abs(transform.det()) == 1
assert abs(reduced.det()) == abs(basis.det())
assert reduced.row(0).norm().n() <= basis.row(0).norm().n()

columns = np.asarray(reduced.list(), dtype=np.int64).reshape(reduced.dimensions()).T
upper = np.linalg.qr(columns, mode="r")
assert is_weakly_lll_reduced(upper, delta)

print("Original basis:")
print(basis)
print("\nDeepLLL-reduced basis:")
print(reduced)
print(f"\nFirst-vector norm: {basis.row(0).norm().n():.6f} -> "
      f"{reduced.row(0).norm().n():.6f}")
print(profile)
print("DEEP_LLL_OK")
