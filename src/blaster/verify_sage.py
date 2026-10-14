"""Sage ``gram_schmidt`` accelerator for the verifier's exact GSO.

Imported only under ``sage -python``. ``sage_gso`` is a drop-in replacement for
``_exact.rational_gso``: it returns the same ``(mu, r)`` in the verifier's
column convention, so a disagreement between the two backends is a verifier
bug, never a reducer verdict.
"""
from fractions import Fraction

from ._exact import rational_gso


def _py_fraction(sage_q):
    """Sage Integer/Rational -> Python Fraction, exactly."""
    return Fraction(int(sage_q.numerator()), int(sage_q.denominator()))


def sage_gso(cols):
    """Exact ``(mu, r)`` via Sage ``matrix(ZZ).gram_schmidt``.

    ``cols`` lists the basis vectors as column vectors. ``matrix(ZZ, cols)``
    places them as *rows*, which is exactly what Sage ``gram_schmidt`` consumes.
    It returns ``(Q, R)`` with ``M = R * Q``, the Gram-Schmidt vectors as the
    rows of ``Q`` (``Q[i] == b*_i``), and ``R[j][i] == mu[i][j]`` for ``i < j``
    (unit diagonal). ``r_i`` and ``mu[i][j]`` are rationals, not ints, so each
    is converted through numerator/denominator.
    """
    from sage.all import matrix, ZZ

    M = matrix(ZZ, cols)  # rows are the basis vectors
    Q, R = M.gram_schmidt(orthonormal=False)
    n = M.nrows()
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    r = [Fraction(0) for _ in range(n)]
    for i in range(n):
        r[i] = _py_fraction(Q[i].dot_product(Q[i]))
        for j in range(i + 1, n):
            mu[i][j] = _py_fraction(R[j, i])
    return mu, r


def resolve_gso(gso):
    """Resolve a GSO backend selector to a ``(mu, r)`` callable.

    ``None`` and ``"rational"`` select the pure-Python exact GSO; ``"sage"``
    selects the Sage ``gram_schmidt`` accelerator; a callable is used directly.
    Both backends return Python ``Fraction`` values under the same column
    convention, so they are interchangeable.
    """
    if gso is None or gso == "rational":
        return rational_gso
    if gso == "sage":
        return sage_gso
    if callable(gso):
        return gso
    raise ValueError("gso must be None, 'rational', 'sage', or a callable")
