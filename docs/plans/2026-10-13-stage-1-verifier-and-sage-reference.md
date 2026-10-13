# Stage 1: Independent Verifier and Sage Reference Reducer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 1 delivery of ADR-0004/ADR-0007 — an exact, independent verifier (`verify_reduction`) and an exact Sage reference reducer (`reference.lll_reduce`) for ordinary LLL with classical panel reduction, both in pure Python with arbitrary-precision integers, plus a Sage `gram_schmidt` accelerator gated behind `sage -python`.

**Architecture:** Three new pure-Python modules under `src/blaster/` hold exact primitives (`_exact.py`), the verifier (`verify.py`), and the reference reducer (`reference.py`); a fourth (`verify_sage.py`) adds an optional Sage GSO accelerator and is never imported by bare Python. The verifier consumes a reducer result `(B0, B, U, delta)` as untrusted data and certifies structure (`B0·U == B`, `abs(det U) == 1`, full column rank) with `int`/`Fraction` arithmetic, then the weak-LLL and size-reduced postconditions with exact rational GSO. The reference reducer transcribes the validated BLASter algorithm (single segment, half-offset toggle, `nearest_core` for local LLL / `nearest_panel` for the panel, two named tie rules, per-operation decision transcript) and never calls Sage `LLL`/`BKZ`.

**Tech Stack:** Python 3.11 (`fractions.Fraction`, `int`), NumPy 2.3 (input conversion only), SageMath 10.6 `gram_schmidt` (accelerator, `sage -python` only), `unittest` (the repo's test framework), Conda env `sage106`.

**Spec:** `docs/spec/stage-1-verifier-and-sage-reference.md` (authoritative contract) and `docs/spec/blaster-algorithm.md` (the `SEGMENTED_LLL` transcription target). Also `docs/adr/0004-separate-reducer-and-verifier.md` and `docs/adr/0007-stage-1-exact-verifier-then-sage-reference.md`.

## Global Constraints

- **Arbitrary-precision only in the verifier/reference**: parse every matrix entry to Python `int`; never use `long long`/NumPy integer types for verifier or reference arithmetic.
- **No floating point in the verifier or reference**: weak-LLL and size-reduced checks are exact rational comparisons; the reference reducer uses exact `Fraction` GSO, no `sqrt`, no `llround`, no `numpy.linalg`.
- **`delta` is the binary64 dyadic**: `delta_to_fraction(delta) == Fraction(*float(delta).as_integer_ratio())`; `float(0.99) != Fraction(99, 100)`. Apply this everywhere `delta` enters exact arithmetic.
- **Column orientation**: `B0`, `B` are matrices whose columns are basis vectors; the relation is `B0 · U == B` with `U` square. The row API `reduce_basis` already handles its own transpose and is out of scope here.
- **Two named tie rules, reproduced exactly**: `nearest_core` (tie away from zero, local LLL) and `nearest_panel` (round-half-to-even, classical panel). No site may substitute Python's built-in `round` for either.
- **Local/global Lovász mismatch preserved**: local LLL uses the non-strict `delta*r_{k-1} <= r_k + mu[k-1,k]^2*r_{k-1}`; the global stop uses the strict weak-LLL `r_i + mu[i,i+1]^2*r_i > delta*r_i`.
- **Sage imports are isolated**: `verify_sage.py` is the only module that imports `sage.all`; nothing in `blaster/__init__.py` may import it. Sage tests live in `tests/sage/` (no `__init__.py`) and run only via `sage -python`.
- **Reference reducer is a transcription, not a Sage call**: it must implement `SEGMENTED_LLL` itself (single segment `segment_size = rank`, half-offset toggle, two-QR structure via exact GSO recompute). Calling `sage.matrix.LLL()`/`BKZ()` as the engine is forbidden.
- **Test framework is `unittest`**, run with `PYTHONPATH=src ... python -m unittest discover ...`. No pytest.
- **Bare-python commands use** `conda run -n sage106 python`; **Sage commands use** `conda run -n sage106 sage -python`. `sage` is not on PATH; `conda` is.
- **Do not commit `t.py`** (untracked scratch at repo root); do not `git add` it.
- **Work on the existing branch `stage-1-verifier-reference`** (already checked out). Do not create a new branch.
- **Git commits end with** `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

The spec states what the software must do but is silent on these five inputs/failure modes most likely to bite a real user. Each gets a test in the owning task.

1. **Exact Lovász equality** — a basis where `r_{i+1} + mu^2 r_i == delta*r_i` exactly must FAIL the strict weak-LLL check (not be rounded across the boundary), while its size-reduced flag still passes. Test: `test_lovasz_equality_boundary_is_strict` (Task 3).
2. **Rank-deficient `B0`** — linearly dependent columns (determinant 0, including a zero column) must raise `VerificationFailed`, not produce a spurious verdict. Test: `test_rank_deficient_raises` (Task 2).
3. **Sage GSO orientation** — `gram_schmidt` on a *non-symmetric* basis must be transposed into the same `(mu, r)` the verifier's column convention expects; a column/row confusion would silently certify the wrong numbers. Test: `test_sage_gso_agrees_with_rational_gso` (Task 4).
4. **`delta` binary64 dyadic** — `0.99` must be interpreted as `float(0.99).as_integer_ratio()`, not `99/100`; a decimal-literal reading would certify a wrong boundary. Test: `test_delta_to_fraction_is_binary64_dyadic` (Task 1).
5. **Entries exceeding signed int64** — a basis with entries `2**60`/`2**90` must parse and certify correctly; an int64 overflow would corrupt the product and determinant checks. Test: `test_wide_dynamic_range_certified` (Task 3).

---

### Task 1: Exact arithmetic primitives (`_exact.py`)

**Files:**
- Create: `src/blaster/_exact.py`
- Test: `tests/test_exact.py`

**Interfaces:**
- Consumes: `src/blaster/_matrix.py` — existing `is_sage_matrix(value)` (returns True when `type(value).__module__.startswith("sage.") and hasattr(value, "base_ring")`).
- Produces (imported by later tasks):
  - `coerce_int(x) -> int` — exact int conversion; raises `TypeError` on float/bool/`Fraction`; raises `ValueError` on non-integral.
  - `to_int_matrix(value, *, name="matrix") -> list[list[int]]` — arbitrary-precision `int` rows from nested lists / NumPy 2-D array / Sage matrix.
  - `det_bareiss(M: list[list[int]]) -> int` — exact determinant (fraction-free Bareiss).
  - `full_column_rank(B0: list[list[int]]) -> bool` — `det_bareiss(gram(B0)) != 0`.
  - `gram(B0) -> list[list[int]]` — `B0^T · B0`.
  - `matmul_columns(B0_rows, U_rows) -> list[list[int]]` — `B0 · U` (columns convention).
  - `column(M_rows, j) -> list[int]`.
  - `transpose_cols(cols) -> list[list[int]]`.
  - `delta_to_fraction(delta) -> Fraction` — `Fraction(*float(delta).as_integer_ratio())`.
  - `rational_gso(cols) -> (mu, r)` — `mu`/`r` lists of `Fraction`.
  - `weak_lll_holds(mu, r, delta: Fraction) -> bool` — strict, all `i`.
  - `is_size_reduced(mu) -> bool` — `abs(mu[i][j]) <= 1/2` for all `i < j`.
  - `nearest_core(x: Fraction) -> int` — tie away from zero.
  - `nearest_panel(x: Fraction) -> int` — round-half-to-even.

- [ ] **Step 1: Write the failing test**

Create `tests/test_exact.py`:

```python
"""Exact arithmetic primitives for the Stage 1 verifier and reference reducer."""
import unittest
from fractions import Fraction

import numpy as np

from blaster._exact import (
    coerce_int,
    column,
    delta_to_fraction,
    det_bareiss,
    full_column_rank,
    is_size_reduced,
    matmul_columns,
    nearest_core,
    nearest_panel,
    rational_gso,
    to_int_matrix,
    weak_lll_holds,
)


class TestCoerceInt(unittest.TestCase):
    def test_accepts_python_int(self):
        self.assertEqual(coerce_int(2**100), 2**100)

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            coerce_int(1.5)
        with self.assertRaises(TypeError):
            coerce_int(1.0)

    def test_rejects_bool(self):
        with self.assertRaises(TypeError):
            coerce_int(True)

    def test_rejects_fraction(self):
        with self.assertRaises(TypeError):
            coerce_int(Fraction(1, 2))


class TestToIntMatrix(unittest.TestCase):
    def test_nested_int_list(self):
        self.assertEqual(to_int_matrix([[1, 2], [3, 4]]), [[1, 2], [3, 4]])

    def test_numpy_int64(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        self.assertEqual(to_int_matrix(a), [[1, 2], [3, 4]])

    def test_large_entries_exceed_int64(self):
        rows = [[1, 2**60], [2**90, -1]]
        self.assertEqual(to_int_matrix(rows), rows)

    def test_rejects_float_entries(self):
        with self.assertRaises(TypeError):
            to_int_matrix([[1.5, 2]])

    def test_rejects_ragged(self):
        with self.assertRaises(ValueError):
            to_int_matrix([[1, 2], [3]])

    def test_rejects_flat_list(self):
        with self.assertRaises(ValueError):
            to_int_matrix([1, 2, 3])


class TestDetBareiss(unittest.TestCase):
    def test_2x2(self):
        self.assertEqual(det_bareiss([[1, 2], [3, 4]]), -2)

    def test_identity(self):
        self.assertEqual(det_bareiss([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1)

    def test_singular(self):
        self.assertEqual(det_bareiss([[1, 2], [2, 4]]), 0)

    def test_permutation_odd(self):
        self.assertEqual(det_bareiss([[0, 1], [1, 0]]), -1)

    def test_known_3x3(self):
        m = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        self.assertEqual(det_bareiss(m), 4)


class TestFullColumnRank(unittest.TestCase):
    def test_full_rank_diagonal(self):
        self.assertTrue(full_column_rank([[1, 0], [0, 1]]))

    def test_dependent_columns(self):
        # columns (1,2) and (2,4) are proportional
        self.assertFalse(full_column_rank([[1, 2], [2, 4]]))

    def test_zero_column(self):
        self.assertFalse(full_column_rank([[1, 0], [0, 0]]))


class TestMatmulColumns(unittest.TestCase):
    def test_identity_transform(self):
        b0 = [[1, 2], [3, 4]]
        u = [[1, 0], [0, 1]]
        self.assertEqual(matmul_columns(b0, u), b0)

    def test_scaling_column(self):
        b0 = [[1, 2], [3, 4]]
        u = [[2, 0], [0, 1]]
        self.assertEqual(matmul_columns(b0, u), [[2, 2], [6, 4]])


class TestDeltaToFraction(unittest.TestCase):
    def test_099_is_binary64_dyadic(self):
        self.assertEqual(delta_to_fraction(0.99),
                         Fraction(*float(0.99).as_integer_ratio()))
        self.assertNotEqual(delta_to_fraction(0.99), Fraction(99, 100))

    def test_exact_dyadics_round_trip(self):
        self.assertEqual(delta_to_fraction(0.5), Fraction(1, 2))
        self.assertEqual(delta_to_fraction(0.75), Fraction(3, 4))


class TestRationalGso(unittest.TestCase):
    def test_2x2(self):
        # columns (2,0) and (1,1): mu[0][1] = 1/2, r = [4, 1]
        mu, r = rational_gso([[2, 0], [1, 1]])
        self.assertEqual(mu[0][1], Fraction(1, 2))
        self.assertEqual(r, [Fraction(4), Fraction(1)])

    def test_identity(self):
        mu, r = rational_gso([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(r, [Fraction(1)] * 3)
        for i in range(3):
            for j in range(3):
                self.assertEqual(mu[i][j], Fraction(0))


class TestWeakLllAndSizeReduced(unittest.TestCase):
    def test_weak_lll_strict(self):
        # r = [4,1], mu[0][1]=1/2: 1 + 1/4*4 = 2 > delta*4 iff delta < 1/2
        mu, r = rational_gso([[2, 0], [1, 1]])
        self.assertTrue(weak_lll_holds(mu, r, Fraction(1, 4)))
        self.assertFalse(weak_lll_holds(mu, r, Fraction(1, 2)))

    def test_size_reduced(self):
        mu, _ = rational_gso([[2, 0], [1, 1]])  # mu = 1/2 is allowed
        self.assertTrue(is_size_reduced(mu))
        mu_bad, _ = rational_gso([[1, 0], [1, 1]])  # mu = 1 is not
        self.assertFalse(is_size_reduced(mu_bad))


class TestNearest(unittest.TestCase):
    def test_nearest_core_ties_away_from_zero(self):
        cases = {
            Fraction(1, 2): 1, Fraction(-1, 2): -1,
            Fraction(3, 2): 2, Fraction(-3, 2): -2,
            Fraction(7, 3): 2, Fraction(-7, 3): -2,
            Fraction(0): 0, Fraction(2): 2,
        }
        for x, want in cases.items():
            with self.subTest(x=x):
                self.assertEqual(nearest_core(x), want)

    def test_nearest_panel_rounds_half_to_even(self):
        cases = {
            Fraction(1, 2): 0, Fraction(-1, 2): 0,
            Fraction(3, 2): 2, Fraction(5, 2): 2, Fraction(7, 2): 4,
            Fraction(-3, 2): -2, Fraction(-5, 2): -2,
            Fraction(1, 3): 0, Fraction(2, 3): 1, Fraction(1): 1,
        }
        for x, want in cases.items():
            with self.subTest(x=x):
                self.assertEqual(nearest_panel(x), want)

    def test_agree_off_ties(self):
        for x in (Fraction(1, 3), Fraction(2, 3), Fraction(1, 4),
                  Fraction(3, 4), Fraction(5, 3), Fraction(-2, 3)):
            self.assertEqual(nearest_core(x), nearest_panel(x))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_exact.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blaster._exact'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/blaster/_exact.py`:

```python
"""Exact arbitrary-precision primitives for the Stage 1 verifier/reference.

All arithmetic here is integer or :class:`fractions.Fraction`; no floating
point, no ``long long``, no NumPy dtype arithmetic. ``B0`` and ``U`` use the
column convention: columns are basis vectors and ``B0 * U == B``.
"""
import operator
from fractions import Fraction

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
    """Normalize ``delta`` to the exact dyadic rational of its binary64 value."""
    return Fraction(*float(delta).as_integer_ratio())


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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_exact.py -v`
Expected: PASS (all tests OK).

- [ ] **Step 5: Commit**

```bash
git add src/blaster/_exact.py tests/test_exact.py
git commit -m "feat(verify): exact integer/rational primitives for Stage 1

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Verifier — structural checks

**Files:**
- Create: `src/blaster/verify.py`
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes from Task 1: `to_int_matrix`, `full_column_rank`, `matmul_columns`, `det_bareiss`, `delta_to_fraction`.
- Produces (used by Tasks 3, 4, 5):
  - `class VerificationFailed(Exception)` — carries `.reason` (str) and `.detail` (str); `__str__` joins them.
  - `class VerificationReport` — frozen dataclass with fields `verified: bool`, `structural_ok: bool`, `full_rank_ok: bool`, `product_ok: bool`, `unimodular_ok: bool`, `weak_lll_ok: bool`, `size_reduced_ok: bool`, `delta: Fraction`, `mode: str`.
  - `verify_reduction(B0, B, U, delta=0.99, *, mode="lll", gso=None) -> VerificationReport` — structural checks raise `VerificationFailed`; postconditions are flags (Task 3 fills the flags).

- [ ] **Step 1: Write the failing test**

Create `tests/test_verifier.py`:

```python
"""Structural checks of the Stage 1 independent verifier."""
import unittest

import numpy as np

from blaster.verify import VerificationFailed, verify_reduction


I2 = [[1, 0], [0, 1]]


class TestStructuralChecks(unittest.TestCase):
    def test_certifies_identity(self):
        rep = verify_reduction(I2, I2, I2, delta=0.99)
        self.assertTrue(rep.verified)
        self.assertTrue(rep.structural_ok)

    def test_product_mismatch_raises(self):
        B = [[1, 0], [0, 2]]
        with self.assertRaises(VerificationFailed):
            verify_reduction(I2, B, I2, delta=0.99)

    def test_nonunimodular_raises(self):
        U = [[2, 0], [0, 1]]
        B = matmul(I2, U)
        with self.assertRaises(VerificationFailed) as cm:
            verify_reduction(I2, B, U, delta=0.99)
        self.assertIn("det", str(cm.exception))

    def test_rank_deficient_raises(self):
        B0 = [[1, 2], [2, 4]]  # columns (1,2),(2,4) dependent
        with self.assertRaises(VerificationFailed):
            verify_reduction(B0, B0, I2, delta=0.99)

    def test_shape_mismatch_raises(self):
        B = [[1, 0, 0], [0, 1, 0]]
        with self.assertRaises(VerificationFailed):
            verify_reduction(I2, B, I2, delta=0.99)

    def test_accepts_numpy_inputs(self):
        a = np.array([[10, 0, 0], [1, 10, 0], [0, 1, 1]], dtype=np.int64)
        rep = verify_reduction(a, a, np.eye(3, dtype=np.int64), delta=0.99)
        self.assertTrue(rep.structural_ok)


def matmul(A, U):
    n = len(A[0])
    p = len(U[0])
    out = [[0] * p for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(p):
            out[i][j] = sum(A[i][k] * U[k][j] for k in range(n))
    return out


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blaster.verify'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/blaster/verify.py` (postcondition flags use Task 1 helpers; the `gso` override and mode validation are here but postcondition flags are exercised in Task 3):

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blaster/verify.py tests/test_verifier.py
git commit -m "feat(verify): structural checks (rank, product, det) with VerificationFailed

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Verifier — reduction postconditions and corruption/corpus tests

**Files:**
- Modify: `src/blaster/__init__.py` (export `verify_reduction`, `VerificationFailed`)
- Test: `tests/test_verifier.py` (append postcondition, corruption, corpus, wide-range tests)

**Interfaces:**
- Consumes from Task 2: `verify_reduction`, `VerificationFailed`; from `blaster.reduce` (`(U, B, tprof) = reduce(A, lll_size=n, delta=0.99, cores=1)`).
- Produces: public exports `blaster.verify_reduction` and `blaster.VerificationFailed`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verifier.py`:

```python
import numpy as np
from blaster import reduce
from blaster.verify import VerificationFailed, verify_reduction


def diag(*xs):
    n = len(xs)
    return [[xs[i] if i == j else 0 for j in range(n)] for i in range(n)]


class TestPostconditions(unittest.TestCase):
    def test_size_reduction_corruption_reported_not_raised(self):
        # row-major [[1,1],[0,1]] means columns (1,0),(1,1): mu[0][1] = 1 > 1/2
        B0 = [[1, 1], [0, 1]]
        rep = verify_reduction(B0, B0, I2, delta=0.99)
        self.assertFalse(rep.size_reduced_ok)
        self.assertTrue(rep.weak_lll_ok)
        self.assertFalse(rep.verified)

    def test_lovasz_equality_boundary_is_strict(self):
        # row-major [[2,1],[0,1]] means columns (2,0),(1,1): r0=4, r1=1, mu=1/2
        # r1 + mu^2 r0 = 1 + 1 = 2 == delta*r0 = 0.5*4 = 2 -> strict fails
        B0 = [[2, 1], [0, 1]]
        rep = verify_reduction(B0, B0, I2, delta=0.5)
        self.assertFalse(rep.weak_lll_ok)
        self.assertTrue(rep.size_reduced_ok)  # mu = 1/2 is allowed
        self.assertFalse(rep.verified)

    def test_wide_dynamic_range_certified(self):
        B0 = diag([1, 2**30, 2**60, 2**90])
        rep = verify_reduction(B0, B0, diag([1, 1, 1, 1]), delta=0.99)
        self.assertTrue(rep.verified)

    def test_delta_reflected_in_report(self):
        rep = verify_reduction(I2, I2, I2, delta=0.99)
        from fractions import Fraction
        self.assertEqual(rep.delta, Fraction(*float(0.99).as_integer_ratio()))


class TestM0Corpus(unittest.TestCase):
    """Acceptance #1: certify the current FastInt64 output on the M0 corpus."""

    CASES = {
        "diag3": np.diag([10, 10, 10]),
        "ex3": np.array([[10, 0, 0], [1, 10, 0], [0, 1, 1]]),
        "ex3b": np.array([[12, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "rand4": np.array([[5, 2, 8, 3], [7, 1, 4, 9], [2, 6, 3, 8], [4, 9, 1, 7]]),
        "rand5": np.array(
            [[3, 7, 1, 9, 2], [8, 4, 6, 5, 1], [9, 2, 7, 4, 8],
             [1, 5, 3, 8, 6], [4, 8, 2, 6, 7]]
        ),
        "rand6": np.random.RandomState(0).randint(-20, 20, size=(6, 6)).astype(np.int64),
    }

    def test_fastint64_output_is_certified(self):
        for name, A in self.CASES.items():
            with self.subTest(name=name):
                n = A.shape[1]
                U, B, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                rep = verify_reduction(A, B, U, delta=0.99)
                self.assertTrue(rep.structural_ok, name)
                self.assertTrue(rep.weak_lll_ok, name)
                self.assertTrue(rep.size_reduced_ok, name)
                self.assertTrue(rep.verified, name)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_verifier.py -v`
Expected: FAIL — the new tests raise `ImportError: cannot import name 'verify_reduction' from 'blaster'` (the export does not exist yet) before the postcondition logic runs.

- [ ] **Step 3: Export the verifier from the package**

Edit `src/blaster/__init__.py` — add the import and the two names to `__all__`:

```python
from .verify import VerificationFailed, verify_reduction
```

and add `"VerificationFailed"` and `"verify_reduction"` to the existing `__all__` list. (The postcondition logic in `verify.py` already exists from Task 2, so no verifier code change is needed here.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_verifier.py -v`
Expected: PASS. If `test_fastint64_output_is_certified` fails on `size_reduced_ok`, confirm the `reduce` call reproduces the M0 baseline first (see debugging note below) before touching verifier code.

Debugging note (only if the corpus test fails): re-run
`PYTHONPATH=src conda run -n sage106 python -c 'import numpy as np; from blaster import reduce; A=np.array([[10,0,0],[1,10,0],[0,1,1]],dtype=np.int64); print(reduce(A, lll_size=3, delta=0.99, cores=1)[:2])'`
and compare against the known M0 output. The verifier's postconditions are exact; a failure here is evidence of a real binary64 decision error (finding #3), not a verifier bug, and must be recorded rather than masked.

- [ ] **Step 5: Run the full bare-python suite**

Run: `make test`
Expected: the existing 24 tests plus the new `test_exact` and `test_verifier` tests all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/__init__.py tests/test_verifier.py
git commit -m "feat(verify): postconditions, corruption rejection, and M0 corpus certification

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: Sage GSO accelerator and `make test-sage`

**Files:**
- Create: `src/blaster/verify_sage.py`
- Create: `tests/sage/test_sage_reference.py`
- Modify: `Makefile` (add `test-sage` target and `SAGE_PYTHON`)
- Modify: `tests/test_api.py` (remove `test_reduce_basis_preserves_sage_matrix_type`; it moves to `tests/sage/`)

**Interfaces:**
- Consumes from Task 1: `rational_gso`, `to_int_matrix` (imported by the test to cross-check and parse; `verify_sage.py` itself imports only `fractions.Fraction` plus a lazy `sage.all`).
- Consumes from Task 2: `verify_reduction` (imported by the test); from `blaster.api`: `reduce_basis` (existing, not created by this plan).
- Produces: `sage_gso(cols) -> (mu, r)` — Sage `gram_schmidt` accelerator; `make test-sage` target.

- [ ] **Step 1: Write the failing Sage test**

Create `tests/sage/test_sage_reference.py` (no `__init__.py` in `tests/sage/`):

```python
"""Sage-only tests: gram_schmidt accelerator and Sage input handling."""
import unittest
from fractions import Fraction

import numpy as np
from sage.all import matrix, ZZ

from blaster._exact import rational_gso, to_int_matrix
from blaster.api import reduce_basis
from blaster.verify import verify_reduction
from blaster.verify_sage import sage_gso


def _cols_of(B_rows):
    m = len(B_rows)
    return [[B_rows[r][j] for r in range(m)] for j in range(len(B_rows[0]))]


class TestSageGso(unittest.TestCase):
    def test_sage_gso_agrees_with_rational_gso(self):
        # Non-symmetric basis: a row/column confusion would disagree here.
        B_rows = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        cols = _cols_of(B_rows)
        mu_s, r_s = sage_gso(cols)
        mu_r, r_r = rational_gso(cols)
        for i in range(len(cols)):
            self.assertEqual(r_s[i], r_r[i])
            for j in range(len(cols)):
                self.assertEqual(mu_s[i][j], mu_r[i][j])

    def test_verify_agrees_with_either_backend(self):
        B0 = [[2, 0], [1, 1]]
        rep_rat = verify_reduction(B0, B0, [[1, 0], [0, 1]], delta=0.99)
        rep_sage = verify_reduction(B0, B0, [[1, 0], [0, 1]], delta=0.99, gso=sage_gso)
        self.assertEqual(rep_rat.verified, rep_sage.verified)
        self.assertEqual(rep_rat.weak_lll_ok, rep_sage.weak_lll_ok)
        self.assertEqual(rep_rat.size_reduced_ok, rep_sage.size_reduced_ok)

    def test_to_int_matrix_accepts_sage(self):
        self.assertEqual(to_int_matrix(matrix(ZZ, [[1, 2], [3, 4]])), [[1, 2], [3, 4]])


class TestReduceBasisSageType(unittest.TestCase):
    def test_reduce_basis_preserves_sage_matrix_type(self):
        A = matrix(ZZ, [[10, 0, 0], [1, 10, 0], [0, 1, 1]])
        B_red, T, _ = reduce_basis(A, return_transform=True, cores=1)
        self.assertTrue(type(B_red).__module__.startswith("sage."))
        self.assertTrue(type(T).__module__.startswith("sage."))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src conda run -n sage106 sage -python -m unittest discover -s tests/sage -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blaster.verify_sage'`.

- [ ] **Step 3: Write the Sage accelerator and move the API test**

Create `src/blaster/verify_sage.py`:

```python
"""Sage ``gram_schmidt`` accelerator for the verifier's exact GSO.

Imported only under ``sage -python``. ``sage_gso`` is a drop-in replacement for
``_exact.rational_gso``: it returns the same ``(mu, r)`` in the verifier's
column convention, so a disagreement between the two backends is a verifier
bug, never a reducer verdict.
"""
from fractions import Fraction


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
```

Edit `tests/test_api.py`: delete the method `test_reduce_basis_preserves_sage_matrix_type` (it now lives in `tests/sage/test_sage_reference.py`). If the method is the last one in its class, leave the class in place; only remove that one method.

- [ ] **Step 4: Run the Sage test to verify it passes**

Run: `PYTHONPATH=src conda run -n sage106 sage -python -m unittest discover -s tests/sage -v`
Expected: PASS.

- [ ] **Step 5: Add the `test-sage` Makefile target**

Edit `Makefile`:
- Add, next to `PYTHON = conda run -n $(CONDA_ENV) python`, the line:
  `SAGE_PYTHON = conda run -n $(CONDA_ENV) sage -python`
- Add `test-sage` to the `.PHONY` line.
- Add the target:

```make
test-sage: cython
	PYTHONPATH=src $(SAGE_PYTHON) -m unittest discover -s tests/sage -v
```

- [ ] **Step 6: Run both suites and confirm bare-python skips Sage tests**

Run: `make test` then `make test-sage`
Expected: `make test` runs bare Python and does **not** attempt `tests/sage/` (no Sage import error); `make test-sage` runs the Sage tests and passes. Confirm `test_reduce_basis_preserves_sage_matrix_type` no longer appears in the bare-python run.

- [ ] **Step 7: Commit**

```bash
git add src/blaster/verify_sage.py tests/sage/test_sage_reference.py tests/test_api.py Makefile
git commit -m "feat(verify): Sage gram_schmidt accelerator and make test-sage target

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: Reference reducer (`reference.py`)

**Files:**
- Create: `src/blaster/reference.py`
- Test: `tests/test_reference.py`

**Interfaces:**
- Consumes from Task 1: `rational_gso`, `nearest_core`, `nearest_panel`, `weak_lll_holds`, `to_int_matrix`, `column`, `transpose_cols`, `delta_to_fraction`.
- Consumes from Task 2: `verify_reduction`.
- Produces: `lll_reduce(B0, delta=0.99) -> (cols, U_rows, transcript)` — the reference reducer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference.py`:

```python
"""Reference reducer: exact BLASter transcription for ordinary LLL."""
import unittest

import numpy as np

from blaster import reduce, verify_reduction
from blaster.reference import lll_reduce


CASES = {
    "diag3": np.diag([10, 10, 10]),
    "ex3": np.array([[10, 0, 0], [1, 10, 0], [0, 1, 1]]),
    "ex3b": np.array([[12, 2, 3], [4, 5, 6], [7, 8, 9]]),
    "rand4": np.array([[5, 2, 8, 3], [7, 1, 4, 9], [2, 6, 3, 8], [4, 9, 1, 7]]),
    "rand5": np.array(
        [[3, 7, 1, 9, 2], [8, 4, 6, 5, 1], [9, 2, 7, 4, 8],
         [1, 5, 3, 8, 6], [4, 8, 2, 6, 7]]
    ),
    "rand6": np.random.RandomState(0).randint(-20, 20, size=(6, 6)).astype(np.int64),
}


class TestReferenceMatchesFastInt64(unittest.TestCase):
    """Acceptance #5: exact B/U equality pins the same deterministic decisions."""

    def test_deterministic_cases_match(self):
        for name, A in CASES.items():
            with self.subTest(name=name):
                n = A.shape[1]
                U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)

    def test_fuzz_matches(self):
        rs = np.random.RandomState(1234)
        for n in range(2, 9):
            for trial in range(20):
                A = rs.randint(-30, 30, size=(n, n)).astype(np.int64)
                try:
                    U_fast, B_fast, _ = reduce(A, lll_size=n, delta=0.99, cores=1)
                except Exception:
                    continue  # rank-deficient or non-square reduce rejected
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                np.testing.assert_array_equal(B_ref, B_fast)
                np.testing.assert_array_equal(np.array(U_ref, dtype=np.int64), U_fast)


class TestReferenceOutputAcceptedByVerifier(unittest.TestCase):
    """Acceptance #4: the reference output is certified by the independent verifier."""

    def test_output_verified(self):
        for name, A in CASES.items():
            with self.subTest(name=name):
                cols_ref, U_ref, _ = lll_reduce(A, delta=0.99)
                B_ref = np.array(cols_ref, dtype=np.int64).T
                rep = verify_reduction(A, B_ref, np.array(U_ref, dtype=np.int64), delta=0.99)
                self.assertTrue(rep.verified, name)

    def test_hnp_basis_certified(self):
        """Acceptance #3: a 61-bit HNP lattice beyond the float64 exact range.

        Findings #2/#3 record float64 rank false-negatives and size-reduction
        errors on 60+ bit HNP lattices. Entries here reach ~2^61 and Gram entries
        ~2^122, so the float fast path (`reduce`) cannot certify them; the exact
        reference reducer plus exact verifier can.
        """
        p = (1 << 61) - 1  # Mersenne prime
        t = [int(x) for x in np.random.RandomState(7).randint(0, p, size=3)]
        B0 = [
            [p, 0, 0, 0],
            [0, p, 0, 0],
            [0, 0, p, 0],
            [t[0], t[1], t[2], 1],
        ]
        cols_ref, U_ref, _ = lll_reduce(B0, delta=0.99)
        B_ref = np.array(cols_ref, dtype=object).T
        rep = verify_reduction(B0, B_ref, np.array(U_ref, dtype=object), delta=0.99)
        self.assertTrue(rep.verified)


class TestTranscript(unittest.TestCase):
    def test_transcript_replays_to_same_result(self):
        A = CASES["rand4"]
        n = A.shape[1]
        rows = A.tolist()
        B = [[rows[r][j] for r in range(n)] for j in range(n)]
        U = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
        cols_ref, U_ref, tr = lll_reduce(A, delta=0.99)
        for rec in tr:
            if rec[0] == "add":
                _, i, k, z, _ = rec
                B[k] = [a + z * b for a, b in zip(B[k], B[i])]
                U[k] = [a + z * b for a, b in zip(U[k], U[i])]
            elif rec[0] == "swap":
                _, a, b = rec
                B[a], B[b] = B[b], B[a]
                U[a], U[b] = U[b], U[a]
            # "offset" records carry no column operation
        self.assertEqual(B, cols_ref)
        # replay accumulates columns; U_ref is row-major, so transpose to compare
        U_rows = [[U[j][r] for j in range(n)] for r in range(n)]
        self.assertEqual(U_rows, U_ref)

    def test_transcript_shape(self):
        cols_ref, U_ref, tr = lll_reduce(CASES["ex3"], delta=0.99)
        self.assertEqual(tr[0][0], "offset")
        self.assertEqual(tr[0][1], 1)  # floor(3/2) = 1 first iteration
        self.assertFalse(tr[0][2])     # first iteration does not stop
        self.assertEqual(tr[-1][0], "offset")
        self.assertTrue(tr[-1][2])     # final iteration reports the stop decision
        for rec in tr:
            self.assertIn(rec[0], ("add", "swap", "offset"))
            if rec[0] == "add":
                self.assertIn(rec[4], ("nearest_core", "nearest_panel"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'blaster.reference'`.

- [ ] **Step 3: Write the reference reducer**

Create `src/blaster/reference.py`:

```python
"""Exact transcription of BLASter ``SEGMENTED_LLL`` for ordinary LLL.

Single segment (``segment_size = rank``), half-offset toggle, classical panel,
exact rational GSO. This is a faithful reproduction of the validated reference
algorithm — it never calls Sage ``LLL``/``BKZ``. The two-QR structure is
realized by recomputing exact GSO after each committed operation; under exact
arithmetic this yields the same ``(mu, r)`` exact Givens would.
"""
from ._exact import (
    column,
    delta_to_fraction,
    nearest_core,
    nearest_panel,
    rational_gso,
    to_int_matrix,
    transpose_cols,
    weak_lll_holds,
)


def _add_col(M, i, k, z):
    """``M[k] += z * M[i]`` in place, where ``M`` is a list of column vectors."""
    M[k] = [a + z * b for a, b in zip(M[k], M[i])]


def _swap_cols(M, a, b):
    M[a], M[b] = M[b], M[a]


def _local_lll(cols, Ucols, offset, delta, tr):
    """Size-reduce with ``nearest_core`` and swap adjacent pairs until local LLL.

    Recomputes GSO after each committed addition: reducing column ``k`` against
    ``b_i`` shifts ``mu[l,k]`` for ``l < i`` by ``z * mu[l,i]``, so a single
    snapshot before the inner loop is wrong.
    """
    n = len(cols)
    k = offset + 1
    while k < n:
        mu, r = rational_gso(cols)
        for i in range(k - 1, offset - 1, -1):
            z = nearest_core(-mu[i][k])
            if z:
                _add_col(cols, i, k, z)
                _add_col(Ucols, i, k, z)
                tr.append(("add", i, k, z, "nearest_core"))
                mu, r = rational_gso(cols)
        mu, r = rational_gso(cols)
        if delta * r[k - 1] <= r[k] + mu[k - 1][k] ** 2 * r[k - 1]:
            k += 1
        else:
            _swap_cols(cols, k - 1, k)
            _swap_cols(Ucols, k - 1, k)
            tr.append(("swap", k - 1, k))
            if k > offset + 1:
                k -= 1


def _size_panel(cols, Ucols, tr):
    """Classical full size-reduction with ``nearest_panel`` (round-half-to-even)."""
    n = len(cols)
    mu, r = rational_gso(cols)
    for k in range(1, n):
        for i in range(k - 1, -1, -1):
            z = nearest_panel(-mu[i][k])
            if z:
                _add_col(cols, i, k, z)
                _add_col(Ucols, i, k, z)
                tr.append(("add", i, k, z, "nearest_panel"))
                mu, r = rational_gso(cols)


def lll_reduce(B0, delta=0.99):
    """Exact segmented-LLL (single segment) returning ``(cols, U_rows, transcript)``.

    ``B0`` is a matrix whose columns are basis vectors (nested lists, NumPy
    2-D array, or Sage matrix). Returns the reduced basis as a list of columns,
    the row-major transform ``U`` with ``B0 * U == B``, and the decision
    transcript.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    m = len(B0_rows)
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)

    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []

    offset = 0
    while True:
        offset = (n // 2) if offset == 0 else 0
        _local_lll(cols, Ucols, offset, d, tr)
        _size_panel(cols, Ucols, tr)
        mu, r = rational_gso(cols)
        stopped = weak_lll_holds(mu, r, d)
        tr.append(("offset", offset, stopped))
        if stopped:
            break

    return cols, transpose_cols(Ucols), tr
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src conda run -n sage106 python -m unittest discover -s tests -p test_reference.py -v`
Expected: PASS. If `test_fuzz_matches` fails, first confirm the M0 corpus cases in `test_deterministic_cases_match` pass; a fuzz-only mismatch on an exact-friendly small-n integer lattice indicates a transcription divergence from the validated algorithm in `_local_lll`/`_size_panel`, not a verifier bug.

- [ ] **Step 5: Run the full bare-python suite**

Run: `make test`
Expected: all tests pass (existing + `test_exact` + `test_verifier` + `test_reference`).

- [ ] **Step 6: Commit**

```bash
git add src/blaster/reference.py tests/test_reference.py
git commit -m "feat(reference): exact single-segment LLL reference reducer with transcript

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Post-plan verification (run once after Task 5)

```bash
make test
make test-sage
git status --short   # confirm only intended files staged; t.py remains untracked
```

Expected: both suites green; the working tree contains exactly the new modules and tests, and `t.py` is present only as an untracked file.
