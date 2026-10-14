"""Exact arbitrary-precision primitives for the Stage 1 verifier/reference.

All arithmetic here is integer or :class:`fractions.Fraction`; no floating
point, no ``long long``, no NumPy dtype arithmetic. ``B0`` and ``U`` use the
column convention: columns are basis vectors and ``B0 * U == B``.
"""
import operator
from fractions import Fraction
from math import isfinite
from numbers import Real

from ._matrix import is_sage_matrix


def coerce_int(x):
    """Return ``x`` as an exact Python ``int``, rejecting lossy types.

    Floats, bools, and Fractions are rejected so a ``1.0`` can never slip
    into an "integer" computation that would later truncate.
    """
    if isinstance(x, bool):
        raise TypeError(f"bool is not an integer: {x!r}")
    if isinstance(x, float):
        raise TypeError(f"float is not an integer: {x!r}")
    if isinstance(x, Fraction):
        raise TypeError(f"Fraction is not an integer: {x!r}")
    try:
        n = operator.index(x)
    except TypeError:
        raise TypeError(f"expected an integer, got {type(x).__name__}") from None
    return int(n)


def to_int_matrix(value, *, name="matrix"):
    """Coerce a nested-list / NumPy 2-D / Sage matrix to ``list[list[int]]`` rows."""
    if is_sage_matrix(value):
        rows = [[int(x) for x in row] for row in value.rows()]
    else:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{name} must be a non-empty 2-D matrix")
        if not all(isinstance(row, (list, tuple)) for row in value):
            raise ValueError(f"{name} must be a 2-D matrix (list of rows)")
        rows = [[coerce_int(x) for x in row] for row in value]
    if not rows:
        raise ValueError(f"{name} must have at least one row")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular and non-empty")
    return rows


def column(M_rows, j):
    """Return column ``j`` of a row-major matrix."""
    return [row[j] for row in M_rows]


def transpose_cols(cols):
    """Transpose a list of column vectors into a row-major matrix."""
    if not cols:
        return []
    return [[c[i] for c in cols] for i in range(len(cols[0]))]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def gram(B0_rows):
    """Gram matrix ``B0^T * B0``."""
    n = len(B0_rows[0])
    return [[_dot(column(B0_rows, i), column(B0_rows, j)) for j in range(n)]
            for i in range(n)]


def det_bareiss(M):
    """Exact determinant via the fraction-free Bareiss algorithm."""
    A = [row[:] for row in M]
    n = len(A)
    if n == 0:
        return 1
    if any(len(row) != n for row in A):
        raise ValueError("determinant requires a square matrix")
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            swap = next((r for r in range(k + 1, n) if A[r][k] != 0), None)
            if swap is None:
                return 0
            A[k], A[swap] = A[swap], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def full_column_rank(B0_rows):
    """True iff ``B0`` has full column rank over Q.

    ``rank(B^T B) == rank(B)`` over Q, and the Gram matrix is square, so
    columns are independent iff ``det(gram(B0)) != 0``. This avoids the
    subtle fraction-free elimination of a rectangular matrix.
    """
    return det_bareiss(gram(B0_rows)) != 0


def matmul_columns(B0_rows, U_rows):
    """Exact product ``B0 * U`` where both arguments are row-major."""
    m = len(B0_rows)
    n = len(B0_rows[0])
    if len(U_rows) != n:
        raise ValueError("U row count must equal B0 column count")
    p = len(U_rows[0])
    cols_B = [column(B0_rows, i) for i in range(n)]
    cols_U = [column(U_rows, j) for j in range(p)]
    out = [[0] * p for _ in range(m)]
    for j in range(p):
        for i in range(n):
            c = cols_U[j][i]
            if c:
                for r in range(m):
                    out[r][j] += cols_B[i][r] * c
    return out


def delta_to_fraction(delta):
    """Normalize ``delta`` to the exact rational used by the exact engines.

    A ``float`` keeps the exact binary64 dyadic of its value, mirroring the
    FastInt64 engine's validation. Exact rational inputs — :class:`Fraction`,
    Sage ``Rational``, or an integer-like value — are preserved exactly instead
    of being rounded through ``float``, so the reference engine never silently
    alters an exact ``delta``. Rejects bools, non-real values, and non-finite
    floats, then enforces the LLL Lovász range ``1/4 < delta < 1`` in exact
    arithmetic.
    """
    if isinstance(delta, bool) or not isinstance(delta, Real):
        raise ValueError("delta must be a finite real number")

    if isinstance(delta, float):
        if not isfinite(delta):
            raise ValueError("delta must be a finite real number")
        d = Fraction(*delta.as_integer_ratio())
    else:
        d = _exact_rational(delta)

    if not Fraction(1, 4) < d < 1:
        raise ValueError("delta must satisfy 0.25 < delta < 1.0")
    return d


def _exact_rational(value):
    """Convert an exact ``Real`` to a :class:`Fraction` without a ``float`` cast.

    Handles :class:`Fraction` and Sage ``Rational`` via numerator/denominator
    (an attribute or a zero-argument method), and integer-like values via
    ``operator.index``. Falls back to binary64 only for exotic ``Real`` types
    that expose neither.
    """
    num = getattr(value, "numerator", None)
    den = getattr(value, "denominator", None)
    if callable(num):
        num = num()
    if callable(den):
        den = den()
    if num is not None and den is not None:
        return Fraction(int(num), int(den))
    try:
        return Fraction(operator.index(value), 1)
    except TypeError:
        f = float(value)
        if not isfinite(f):
            raise ValueError("delta must be a finite real number")
        return Fraction(*f.as_integer_ratio())


def rational_gso(cols):
    """Exact Gram-Schmidt of column vectors.

    Returns ``(mu, r)`` where ``mu[i][j]`` is the GSO coefficient of column
    ``j`` onto ``b*_i`` for ``i < j`` (and ``0`` otherwise), and ``r[i]`` is
    ``<b*_i, b*_i>``. The recursion follows the Gram-matrix LDL^T identity:
    ``r_i = <b_i,b_i> - sum_{k<i} mu[k,i]^2 r_k`` and
    ``mu[i,j] = (<b_i,b_j> - sum_{k<i} mu[k,i] mu[k,j] r_k) / r_i``.
    """
    n = len(cols)
    G = [[_dot(cols[i], cols[j]) for j in range(n)] for i in range(n)]
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    r = [Fraction(0) for _ in range(n)]
    for i in range(n):
        s = Fraction(G[i][i])
        for k in range(i):
            s -= mu[k][i] ** 2 * r[k]
        r[i] = s
        for j in range(i + 1, n):
            t = Fraction(G[i][j])
            for k in range(i):
                t -= mu[k][i] * mu[k][j] * r[k]
            mu[i][j] = t / r[i]
    return mu, r


def weak_lll_holds(mu, r, delta):
    """Strict weak-LLL(delta): ``r_{i+1} + mu[i,i+1]^2 r_i > delta r_i``."""
    for i in range(len(r) - 1):
        if not r[i + 1] + mu[i][i + 1] ** 2 * r[i] > delta * r[i]:
            return False
    return True


def is_size_reduced(mu):
    """Non-strict size-reduced: ``abs(mu[i][j]) <= 1/2`` for all ``i < j``."""
    n = len(mu)
    for i in range(n):
        for j in range(i + 1, n):
            if 2 * abs(mu[i][j]) > 1:
                return False
    return True


def deep_lll_holds(mu, r, delta, depth):
    """Strict deep-LLL(delta, depth): the deep condition over the window.

    For every ``k`` and every in-window ``i < k`` (``i < depth or
    i >= k - depth``), ``||pi_i(b_k)||^2 > delta * r_i``, where
    ``||pi_i(b_k)||^2 = r_k + sum_{t=i}^{k-1} mu[t][k]^2 r_t``. ``rational_gso``
    stores ``mu[k][k] = 0``, so the diagonal term is ``r_k`` itself; the sum is
    maintained incrementally as ``proj -= mu[i][k]^2 r_i`` after each test.
    """
    n = len(r)
    for k in range(1, n):
        proj = r[k] + sum(mu[t][k] ** 2 * r[t] for t in range(k))
        for i in range(k):
            if i < depth or i >= k - depth:
                if not proj > delta * r[i]:
                    return False
            proj -= mu[i][k] ** 2 * r[i]
    return True


def nearest_core(x):
    """Nearest integer to the rational ``x``, ties away from zero.

    ``sign(a) * floor((2*abs(a) + b) / (2*b))`` for ``x = a/b`` with ``b > 0``.
    """
    a, b = x.numerator, x.denominator
    if a < 0:
        return -((2 * (-a) + b) // (2 * b))
    return (2 * a + b) // (2 * b)


def nearest_panel(x):
    """Nearest integer to the rational ``x``, round-half-to-even.

    On an exact half-integer the magnitude rounds to the nearest even integer;
    otherwise it rounds to the nearest integer.
    """
    a, b = x.numerator, x.denominator
    q, rem = divmod(a, b)
    tw = 2 * rem
    if tw < b:
        return q
    if tw > b:
        return q + 1
    return q if q % 2 == 0 else q + 1
