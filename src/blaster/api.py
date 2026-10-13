"""High-level, row-oriented lattice reduction API."""

from ._matrix import as_int64_matrix, is_sage_matrix, to_sage_matrix
from .blaster import reduce


def reduce_basis(
    basis,
    *,
    orientation="rows",
    return_transform=False,
    **parameters,
):
    """Reduce an integer lattice basis represented by rows or columns.

    ``orientation="rows"`` is the default expected by SageMath and most lattice
    tools. Set ``orientation="columns"`` to use BLASter's internal convention.
    The result has the same orientation and, for a Sage input, is returned as a
    Sage integer matrix.

    When ``return_transform`` is true, return ``(reduced, transform, profile)``.
    For row bases, ``transform @ basis == reduced``. For column bases,
    ``basis @ transform == reduced``.
    """
    if orientation not in {"rows", "columns"}:
        raise ValueError("orientation must be 'rows' or 'columns'")

    sage_input = is_sage_matrix(basis)
    sage_ring = basis.base_ring() if sage_input else None
    array = as_int64_matrix(basis, name="basis")
    columns = array.T.copy() if orientation == "rows" else array

    transform, reduced_columns, profile = reduce(columns, **parameters)
    if orientation == "rows":
        reduced = reduced_columns.T.copy()
        public_transform = transform.T.copy()
    else:
        reduced = reduced_columns
        public_transform = transform

    if sage_input:
        reduced = to_sage_matrix(reduced, sage_ring)
        public_transform = to_sage_matrix(public_transform)

    if return_transform:
        return reduced, public_transform, profile
    return reduced
