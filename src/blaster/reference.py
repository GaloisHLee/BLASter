"""Exact transcription of BLASter ``SEGMENTED_LLL`` for ordinary LLL.

Segmented by a ``lll_size`` (default: the whole rank), half-offset toggle,
classical panel, exact rational GSO. This is a faithful reproduction of the
validated reference algorithm — it never calls Sage ``LLL``/``BKZ``. The
two-QR structure is realized by recomputing exact GSO after each committed
operation; under exact arithmetic this yields the same ``(mu, r)`` exact
Givens would.
"""
from ._exact import (
    coerce_int,
    column,
    delta_to_fraction,
    full_column_rank,
    nearest_core,
    nearest_panel,
    to_int_matrix,
    transpose_cols,
    weak_lll_holds,
)
from .verify_sage import resolve_gso


def _add_col(M, i, k, z):
    """``M[k] += z * M[i]`` in place, where ``M`` is a list of column vectors."""
    M[k] = [a + z * b for a, b in zip(M[k], M[i])]


def _swap_cols(M, a, b):
    M[a], M[b] = M[b], M[a]


def _local_lll(cols, Ucols, start, end, delta, tr, gso):
    """Size-reduce with ``nearest_core`` and swap adjacent pairs until local LLL.

    Reduces only the columns ``[start, end)``; columns outside the segment are
    untouched. Recomputes GSO after each committed addition: reducing column
    ``k`` against ``b_i`` shifts ``mu[l,k]`` for ``l < i`` by ``z * mu[l,i]``,
    so a single snapshot before the inner loop is wrong.
    """
    k = start + 1
    while k < end:
        mu, r = gso(cols)
        for i in range(k - 1, start - 1, -1):
            z = nearest_core(-mu[i][k])
            if z:
                _add_col(cols, i, k, z)
                _add_col(Ucols, i, k, z)
                tr.append(("add", i, k, z, "nearest_core"))
                mu, r = gso(cols)
        mu, r = gso(cols)
        if delta * r[k - 1] <= r[k] + mu[k - 1][k] ** 2 * r[k - 1]:
            k += 1
        else:
            _swap_cols(cols, k - 1, k)
            _swap_cols(Ucols, k - 1, k)
            tr.append(("swap", k - 1, k))
            if k > start + 1:
                k -= 1


def _size_panel(cols, Ucols, tr, gso):
    """Classical full size-reduction with ``nearest_panel`` (round-half-to-even)."""
    n = len(cols)
    mu, r = gso(cols)
    for k in range(1, n):
        for i in range(k - 1, -1, -1):
            z = nearest_panel(-mu[i][k])
            if z:
                _add_col(cols, i, k, z)
                _add_col(Ucols, i, k, z)
                tr.append(("add", i, k, z, "nearest_panel"))
                mu, r = gso(cols)


def _segment_width(lll_size, n):
    """Normalize ``lll_size`` to an effective segment width in ``[2, n]``.

    ``None`` means a single segment spanning the whole rank. A value smaller
    than 2 is rejected (mirroring the FastInt64 engine's minimum); a value at
    or above ``n`` collapses to one segment. ``coerce_int`` keeps a ``float``,
    ``bool``, or :class:`~fractions.Fraction` from slipping through.
    """
    if lll_size is None:
        return n
    w = coerce_int(lll_size)
    if w < 2:
        raise ValueError("lll_size must be at least 2")
    return min(w, n)


def lll_reduce(B0, delta=0.99, *, gso=None, lll_size=None):
    """Exact segmented-LLL returning ``(cols, U_rows, transcript)``.

    ``B0`` is a matrix whose columns are basis vectors (nested lists, NumPy
    2-D array, or Sage matrix). Returns the reduced basis as a list of columns,
    the row-major transform ``U`` with ``B0 * U == B``, and the decision
    transcript.

    ``lll_size`` is the segment width. When it is less than the rank ``n``, the
    columns ``[offset, n)`` are partitioned into disjoint ``lll_size``-wide
    segments and each is reduced locally in a deterministic sequential order
    (a faithful sequential transcription of FastInt64's parallel ``block_lll``).
    The default (``None``) is a single segment covering the whole rank.

    ``gso`` selects the exact Gram-Schmidt primitive the reducer uses at every
    step: ``None``/``"rational"`` for pure Python, ``"sage"`` for Sage's
    ``gram_schmidt``, or a callable returning ``(mu, r)``. The algorithm itself
    is unchanged — only the exact GSO backend is swapped, and it never calls
    Sage ``LLL``/``BKZ``.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    if not full_column_rank(B0_rows):
        raise ValueError("basis vectors must be linearly independent")
    m = len(B0_rows)
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    gso_fn = resolve_gso(gso)
    block = _segment_width(lll_size, n)

    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []

    offset = 0
    while True:
        offset = (block // 2) if offset == 0 else 0
        # The offset record heads each iteration; a closing offset record with
        # stopped=True is appended once the final iteration proves weakly-LLL
        # reduced, so no add/swap record is ever clobbered.
        tr.append(("offset", offset, False))
        for i in range(offset, n, block):
            _local_lll(cols, Ucols, i, min(i + block, n), d, tr, gso_fn)
        _size_panel(cols, Ucols, tr, gso_fn)
        mu, r = gso_fn(cols)
        if weak_lll_holds(mu, r, d):
            tr.append(("offset", offset, True))
            break

    return cols, transpose_cols(Ucols), tr


def lll_reduce_rows(A0, delta=0.99, *, gso=None, lll_size=None):
    """Row-convention entry point: ``A0`` rows are the basis vectors.

    Returns ``(rows, T, transcript)`` with ``T @ A0 == rows``. This is the
    transpose of the column convention: reduce ``A0^T``, then transpose the
    reduced basis and the transform back (``rows == B^T``, ``T == U^T``).
    ``gso`` and ``lll_size`` are forwarded to :func:`lll_reduce`.
    """
    A0_rows = to_int_matrix(A0, name="A0")
    cols, U_rows, tr = lll_reduce(
        transpose_cols(A0_rows), delta, gso=gso, lll_size=lll_size
    )
    rows = [c[:] for c in cols]  # reduced basis vectors as rows (== B^T)
    T = transpose_cols(U_rows)   # == U^T
    return rows, T, tr
