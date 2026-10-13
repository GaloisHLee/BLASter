"""Independent exact verifier for Stage 1 (LLL + classical panel).

Consumes a reducer result ``(B0, B, U, delta)`` as untrusted data and certifies
the structural and reduction contracts with exact integer/rational arithmetic.
It has no access to reducer scratch state and trusts no reducer claim.
"""
from dataclasses import dataclass
from fractions import Fraction

from ._exact import (
    delta_to_fraction,
    det_bareiss,
    full_column_rank,
    is_size_reduced,
    matmul_columns,
    rational_gso,
    to_int_matrix,
    weak_lll_holds,
)


class VerificationFailed(Exception):
    """A structural check failed; the result is left unverified."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail

    def __str__(self):
        return self.reason if not self.detail else f"{self.reason}: {self.detail}"


@dataclass(frozen=True)
class VerificationReport:
    verified: bool
    structural_ok: bool
    full_rank_ok: bool
    product_ok: bool
    unimodular_ok: bool
    weak_lll_ok: bool
    size_reduced_ok: bool
    delta: Fraction
    mode: str


def verify_reduction(B0, B, U, delta=0.99, *, mode="lll", gso=None):
    """Verify a reduction result and return a per-check ``VerificationReport``.

    Raises ``VerificationFailed`` on any failed structural check. Postcondition
    failures are reported as flags on the returned report, not raised.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    B_rows = to_int_matrix(B, name="B")
    U_rows = to_int_matrix(U, name="U")

    m = len(B0_rows)
    n = len(B0_rows[0])
    if len(U_rows) != n or len(U_rows[0]) != n:
        raise VerificationFailed("U must be square with side equal to B0 column count")
    if len(B_rows) != m or len(B_rows[0]) != n:
        raise VerificationFailed(
            "B shape does not match B0", f"B0 {m}x{n} vs B {len(B_rows)}x{len(B_rows[0])}"
        )

    if not full_column_rank(B0_rows):
        raise VerificationFailed("B0 does not have full column rank over Q")

    product = matmul_columns(B0_rows, U_rows)
    if product != B_rows:
        raise VerificationFailed("B0 * U != B")

    if abs(det_bareiss(U_rows)) != 1:
        raise VerificationFailed("det(U) must be +/-1", f"det(U) = {det_bareiss(U_rows)}")

    d = delta_to_fraction(delta)
    cols = [[B_rows[r][j] for r in range(m)] for j in range(n)]
    mu, r = (gso or rational_gso)(cols)

    weak_lll_ok = weak_lll_holds(mu, r, d)
    size_reduced_ok = is_size_reduced(mu)

    return VerificationReport(
        verified=weak_lll_ok and size_reduced_ok,
        structural_ok=True,
        full_rank_ok=True,
        product_ok=True,
        unimodular_ok=True,
        weak_lll_ok=weak_lll_ok,
        size_reduced_ok=size_reduced_ok,
        delta=d,
        mode=mode,
    )
