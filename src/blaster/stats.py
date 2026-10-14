"""
Utility functions for computing the basis profile of a given basis, and measuring its quality.
"""
from fractions import Fraction
from math import exp, gamma, log2, pi
import numpy as np


def get_profile(basis, is_upper=False, *, gso=None):
    """
    Return the profile of a basis, i.e. log_2 ||b_i*|| for i=1, ..., n.
    Note: the logarithm is done base 2, similar to https://github.com/keeganryan/flatter.
    :param basis: basis for a lattice
    :param is_upper: whether `basis` is already an upper triangular matrix or not
    :param gso: exact Gram-Schmidt backend selector for object-dtype bases
        (``None``/``"rational"`` for pure-Python :class:`Fraction`, or
        ``"sage"`` for Sage's accelerator)
    """
    if is_upper:
        upper = basis
    else:
        array = np.asarray(basis)
        if array.dtype.kind == "O":
            return _object_profile(basis, gso=gso)
        upper = np.linalg.qr(basis, mode='r')
    return [log2(abs(d_i)) for d_i in upper.diagonal()]


def _log2_exact(value):
    """Return ``log2`` of an exact ``int``/``Fraction`` without a float cast.

    ``np.linalg.qr`` coerces object entries to ``float64``, which overflows or
    rounds for entries beyond that range. ``math.log2`` takes an integer
    directly, and a :class:`Fraction` is handled via numerator/denominator so
    neither overflows.
    """
    if isinstance(value, Fraction):
        return log2(value.numerator) - log2(value.denominator)
    return log2(value)


def _object_profile(basis, *, gso=None):
    """Exact log2 Gram-Schmidt norms of a column-oriented object-dtype basis.

    Matches :func:`get_profile` without ``np.linalg.qr``, which cannot handle
    arbitrary-precision (object-dtype) entries. Uses the exact GSO backend
    selected by ``gso`` (same contract as :func:`blaster.verify.verify_reduction`).
    """
    from .verify_sage import resolve_gso

    columns = [list(basis[:, j]) for j in range(basis.shape[1])]
    _, r = resolve_gso(gso)(columns)
    return [0.5 * _log2_exact(ri) for ri in r]


def gh(dim):
    """
    Return the Gaussian Heuristic at dimension `dim`. This gives a prediction of
    the length of the shortest vector in a lattice of unit volume.
    :param dim: lattice dimension
    :return: GH(dim)
    """
    if dim >= 100:
        return float(dim / (2*pi*exp(1)))**0.5
    return float(gamma(1.0 + 0.5 * dim)**(1.0 / dim) / pi**0.5)


def gaussian_heuristic(profile):
    """
    Return the Gaussian Heuristic for a particular basis profile.
    """
    n = len(profile)
    return gh(n) * 2.0**(sum(profile) / n)


def rhf(profile):
    """
    Return the n-th root Hermite factor, given a basis profile, i.e.,
        rhf(B) = (||b_0|| / det(B)^{1/n})^{1/n}.
    """
    n = len(profile)
    return 2.0**((profile[0] - sum(profile) / n) / n)


def slope(profile):
    """
    Return the current slope of a basis profile
    """
    n = len(profile)
    if n < 2:
        return 0.0
    i_mean = (n - 1) * 0.5
    v1 = sum(profile[i] * (i - i_mean) for i in range(n))
    v2 = sum((i - i_mean)**2 for i in range(n))
    return v1 / v2


def potential(profile):
    """
    Return the (log2 of the) potential of a basis profile.
    Normally in lattice reduction, this is a strictly decreasing function of time, and is used to
    prove that LLL runs in polynomial time.
    """
    n = len(profile)
    return sum((n - i) * profile[i] for i in range(n))
