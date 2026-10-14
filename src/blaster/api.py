"""High-level, row-oriented lattice reduction API."""

import numpy as np

from ._exact import to_int_matrix, transpose_cols
from ._matrix import as_int64_matrix, is_sage_matrix, to_sage_matrix
from .blaster import reduce
from .reference import lll_reduce, lll_reduce_rows


def reduce_basis(
    basis,
    *,
    orientation="rows",
    engine="fast",
    return_transform=False,
    **parameters,
):
    """Reduce an integer lattice basis represented by rows or columns.

    ``orientation="rows"`` is the default expected by SageMath and most lattice
    tools. Set ``orientation="columns"`` to use BLASter's internal convention.
    The result has the same orientation and, for a Sage input, is returned as a
    Sage integer matrix.

    ``engine`` selects the backend:

    * ``"fast"`` (default) — the C-accelerated FastInt64 engine, which requires
      every entry to fit in signed int64 and computes GSO in binary64.
    * ``"reference"`` — the exact arbitrary-precision BLASter transcription
      (ordinary LLL only). It accepts entries of any magnitude, supports the
      ``delta``, ``gso``, ``lll_size`` and ``use_seysen`` parameters, and its third return value
      is the decision transcript (a list) rather than a ``TimeProfile``. ``gso``
      selects the exact Gram-Schmidt backend: ``"rational"`` (default) or
      ``"sage"``. ``lll_size`` is the segment width (default: whole rank);
      ``use_seysen=True`` selects the recursive Seysen panel instead of
      classical size reduction.

    When ``return_transform`` is true, return ``(reduced, transform, extra)``.
    For row bases, ``transform @ basis == reduced``. For column bases,
    ``basis @ transform == reduced``. ``extra`` is a ``TimeProfile`` for the
    fast engine and a decision transcript for the reference engine.
    """
    if orientation not in {"rows", "columns"}:
        raise ValueError("orientation must be 'rows' or 'columns'")
    if engine not in {"fast", "reference"}:
        raise ValueError("engine must be 'fast' or 'reference'")
    if engine == "reference":
        return _reduce_basis_reference(
            basis, orientation=orientation, return_transform=return_transform, **parameters
        )

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


def _reduce_basis_reference(basis, *, orientation, return_transform, **parameters):
    """Exact arbitrary-precision LLL path of :func:`reduce_basis`."""
    delta = parameters.pop("delta", 0.99)
    gso = parameters.pop("gso", None)
    lll_size = parameters.pop("lll_size", None)
    use_seysen = parameters.pop("use_seysen", False)
    if parameters:
        raise ValueError(
            "reference engine supports only 'delta', 'gso', 'lll_size' and "
            "'use_seysen', got: " + ", ".join(sorted(parameters))
        )

    sage_input = is_sage_matrix(basis)
    sage_ring = basis.base_ring() if sage_input else None
    rows = to_int_matrix(basis, name="basis")

    if orientation == "rows":
        reduced_rows, transform, transcript = lll_reduce_rows(
            rows, delta=delta, gso=gso, lll_size=lll_size, use_seysen=use_seysen
        )
    else:
        cols, U_rows, transcript = lll_reduce(
            rows, delta=delta, gso=gso, lll_size=lll_size, use_seysen=use_seysen
        )
        reduced_rows = transpose_cols(cols)
        transform = U_rows

    reduced = np.array(reduced_rows, dtype=object)
    transform = np.array(transform, dtype=object)
    if sage_input:
        reduced = to_sage_matrix(reduced, sage_ring)
        transform = to_sage_matrix(transform)

    if return_transform:
        return reduced, transform, transcript
    return reduced
