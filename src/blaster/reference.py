"""Exact transcription of BLASter ``SEGMENTED_LLL`` for ordinary LLL.

Segmented by a ``lll_size`` (default: the whole rank), half-offset toggle,
classical panel, exact rational GSO. This is a faithful reproduction of the
validated reference algorithm — it never calls Sage ``LLL``/``BKZ``. The
two-QR structure is realized by recomputing exact GSO after each committed
operation; under exact arithmetic this yields the same ``(mu, r)`` exact
Givens would.
"""
from fractions import Fraction
from functools import cache
from numbers import Integral

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


def _deeplll_local(cols, Ucols, start, end, delta, depth, tr, gso):
    """Depth-deep-LLL over ``[start, end)``: local LLL then deep insertions.

    Transcribes FastInt64 ``_deeplll_reduce``: after ``_local_lll``, walk
    ``k`` and deep-insert ``b_k`` at the earliest ``i < k`` satisfying the
    depth predicate ``(i < depth or i >= k - depth)`` and
    ``||pi_i(b_k)||^2 <= delta * ||b_i*||^2``, via adjacent swaps. The
    ``("deep", i, k)`` record (global indices) marks the decision; the actual
    column movement is the following ``("swap", ...)`` run.
    """
    _local_lll(cols, Ucols, start, end, delta, tr, gso)
    k = 1
    while k < end - start:
        mu, r = gso(cols)
        for i in range(k - 1, -1, -1):
            z = nearest_core(-mu[start + i][start + k])
            if z:
                _add_col(cols, start + i, start + k, z)
                _add_col(Ucols, start + i, start + k, z)
                tr.append(("add", start + i, start + k, z, "nearest_core"))
                mu, r = gso(cols)
        mu, r = gso(cols)
        # ||b_k||^2 = r[k] + sum_{t<k} mu[t][k]^2 r[t]: rational_gso sets
        # mu[k][k] = 0, so the diagonal term is r[k] itself.
        proj = r[start + k] + sum(
            mu[start + t][start + k] ** 2 * r[start + t] for t in range(k)
        )
        swapped = False
        for i in range(k):
            if (i < depth or i >= k - depth) and proj <= delta * r[start + i]:
                tr.append(("deep", start + i, start + k))
                while k > i:
                    _swap_cols(cols, start + k - 1, start + k)
                    _swap_cols(Ucols, start + k - 1, start + k)
                    tr.append(("swap", start + k - 1, start + k))
                    k -= 1
                if k == 0:
                    k += 1
                swapped = True
                break
            proj -= mu[start + i][start + k] ** 2 * r[start + i]
        if not swapped:
            k += 1


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


def _matmul(A, B):
    """Exact matrix product of two lists of rows."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    if ca != rb:
        raise ValueError("matrix dimension mismatch")
    return [
        [sum(A[i][t] * B[t][j] for t in range(ca)) for j in range(cb)]
        for i in range(ra)
    ]


def _solve_unit_upper(M, X):
    """Solve ``M * Y == X`` for a unit upper-triangular ``M`` by back-substitution.

    The unit diagonal means no division is needed; ``Y = M^{-1} X`` is computed
    exactly over :class:`fractions.Fraction`.
    """
    m = len(M)
    Y = [row[:] for row in X]
    for i in range(m - 1, -1, -1):
        for t in range(i + 1, m):
            if M[i][t]:
                for j in range(len(X[0])):
                    Y[i][j] -= M[i][t] * Y[t][j]
    return Y


@cache
def _reduction_ranges(n):
    """Non-recursive port of FastInt64 ``size_reduction.__reduction_ranges``.

    Returns ``(base_cases, ranges)``: ``base_cases`` lists indices ``i`` whose
    pair ``(i, i+1)`` forms a two-column Seysen leaf, and ``ranges`` lists the
    ``(i, j, k)`` triples where ``[j:k)`` is Seysen-reduced against ``[i:j)``.
    """
    bit_shift, parts, result, base_cases = 1, 1, [], []
    while parts < n:
        left_bound, left_idx = 0, 0
        for i in range(1, parts + 1):
            right_bound = left_bound + 2 * n
            mid_idx = (left_bound + n) >> bit_shift
            right_idx = right_bound >> bit_shift
            if right_idx > left_idx + 1:
                if right_idx == left_idx + 2:
                    base_cases.append(left_idx)
                else:
                    result.append((left_idx, mid_idx, right_idx))
            left_bound, left_idx = right_bound, right_idx
        parts *= 2
        bit_shift += 1
    return base_cases, list(reversed(result))


def _seysen_panel(cols, Ucols, tr, gso):
    """Recursive Seysen panel (two-column leaf + rounded-inverse cross-panel).

    Transcribes FastInt64 ``seysen_reduce`` exactly, on the unit-upper-triangular
    matrix ``M`` (``M[a][b] = mu[a][b]`` for ``a < b``) rather than on the QR
    factor ``R = D * M``. ``D = diag(sqrt(r_i))`` is invariant under Seysen
    (adding lower-index columns to higher ones preserves every ``b*_i``), so it
    cancels out of every rounding decision and every in-place update. As in the
    FastInt64 engine, a base case writes only ``M[i][i+1]`` and leaves
    ``M[l][i+1]`` (``l < i``) stale for the cross-panel ranges; the transform is
    accumulated in ``U`` and applied to the columns once, at the end.
    """
    n = len(cols)
    mu, _ = gso(cols)
    M = [
        [Fraction(1) if a == b else (mu[a][b] if a < b else Fraction(0)) for b in range(n)]
        for a in range(n)
    ]
    U = [[Fraction(1) if a == b else Fraction(0) for b in range(n)] for a in range(n)]

    base_cases, ranges = _reduction_ranges(n)
    for i in base_cases:
        z = nearest_panel(-M[i][i + 1])
        U[i][i + 1] = z
        M[i][i + 1] += z
    for (i, j, k) in ranges:
        M11 = [row[i:j] for row in M[i:j]]
        M12 = [row[j:k] for row in M[i:j]]
        U22 = [row[j:k] for row in U[j:k]]
        U11 = [row[i:j] for row in U[i:j]]
        M12 = _matmul(M12, U22)                                       # S12' = M12 * U22
        X = _solve_unit_upper(M11, [[-x for x in row] for row in M12])  # -M11^{-1} * M12
        U12p = [[nearest_panel(x) for x in row] for row in X]
        M12 = [
            [M12[a][b] + v for b, v in enumerate(row)]
            for a, row in enumerate(_matmul(M11, U12p))
        ]                                                            # S12 = S12' + M11 * U12p
        U12 = _matmul(U11, U12p)
        for a in range(j - i):
            for b in range(k - j):
                M[i + a][j + b] = M12[a][b]
                U[i + a][j + b] = U12[a][b]

    for b in range(n - 1, -1, -1):
        for a in range(b):
            z = U[a][b]
            if z:
                zz = int(z)
                _add_col(cols, a, b, zz)
                _add_col(Ucols, a, b, zz)
                tr.append(("add", a, b, zz, "nearest_panel"))


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


def _depth_param(depth):
    """Validate ``depth`` exactly as the FastInt64 engine does.

    ``bool`` and non-integral values are rejected (``Integral`` covers both
    Python ``int`` and Sage ``Integer``); the value is then range-checked.
    """
    if isinstance(depth, bool) or not isinstance(depth, Integral):
        raise ValueError("depth must be an integer")
    depth = int(depth)
    if depth < 0:
        raise ValueError("depth must be at least 0")
    return depth


def _progressive_betas(beta, increment):
    """Return the upstream progressive-beta schedule for the validated block size.

    Exact transcription of FastInt64 ``_progressive_betas`` (pure integer
    arithmetic, no floating point). ``beta`` is the validated block size and
    ``increment`` is ``bkz_prog`` (``1 <= increment <= beta``):

    * ``beta < 40`` selects DeepLLL-4 preprocessing rather than a pruned BKZ
      tour, so the schedule is empty.
    * ``increment >= beta`` collapses to the single final tour ``(beta,)``.
    * otherwise the schedule is ``range(start, beta + 1, increment)`` with
      ``start = 40 + ((beta - 40) % increment)`` — the smallest value ``>= 40``
      congruent to ``beta`` modulo ``increment`` — so it always begins in
      ``[40, 40 + increment)`` and always ends at ``beta``.
    """
    if beta < 40:
        return ()
    if increment >= beta:
        return (beta,)
    start = 40 + ((beta - 40) % increment)
    return tuple(range(start, beta + 1, increment))


def lll_reduce(B0, delta=0.99, *, gso=None, lll_size=None, use_seysen=False, depth=0):
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

    ``depth`` (default ``0``) selects the deep-LLL variant: when truthy, each
    segment is reduced by :func:`_deeplll_local` (local LLL followed by deep
    insertions) instead of :func:`_local_lll`. ``depth=0`` is exactly ordinary
    segmented LLL.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    if not full_column_rank(B0_rows):
        raise ValueError("basis vectors must be linearly independent")
    m = len(B0_rows)
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    gso_fn = resolve_gso(gso)
    block = _segment_width(lll_size, n)
    depth = _depth_param(depth)

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
            w = min(i + block, n) - i
            tr.append(("segment", i, w))
            if depth:
                _deeplll_local(cols, Ucols, i, min(i + block, n), d, depth, tr, gso_fn)
            else:
                _local_lll(cols, Ucols, i, min(i + block, n), d, tr, gso_fn)
        (_seysen_panel if use_seysen else _size_panel)(cols, Ucols, tr, gso_fn)
        mu, r = gso_fn(cols)
        if weak_lll_holds(mu, r, d):
            tr.append(("offset", offset, True))
            break

    return cols, transpose_cols(Ucols), tr


def lll_reduce_rows(A0, delta=0.99, *, gso=None, lll_size=None, use_seysen=False, depth=0):
    """Row-convention entry point: ``A0`` rows are the basis vectors.

    Returns ``(rows, T, transcript)`` with ``T @ A0 == rows``. This is the
    transpose of the column convention: reduce ``A0^T``, then transpose the
    reduced basis and the transform back (``rows == B^T``, ``T == U^T``).
    ``gso``, ``lll_size``, ``use_seysen`` and ``depth`` are forwarded to
    :func:`lll_reduce`.
    """
    A0_rows = to_int_matrix(A0, name="A0")
    cols, U_rows, tr = lll_reduce(
        transpose_cols(A0_rows), delta, gso=gso, lll_size=lll_size,
        use_seysen=use_seysen, depth=depth,
    )
    rows = [c[:] for c in cols]  # reduced basis vectors as rows (== B^T)
    T = transpose_cols(U_rows)   # == U^T
    return rows, T, tr
