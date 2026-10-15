# BKZ full-tour transcription + progressive-schedule wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the reference engine's BKZ coverage — promote its per-segment kernel into a full progressive/segmented multi-tour driver, and wire `_progressive_betas` into the reference reducer's runtime (scheduler + public API).

**Architecture:** Three faithful layers are transcribed: `_bkz_segment` (C++ inner kernel, one segment), `bkz_reduce` (FastInt64 tour driver: pre-LOLL → segmented multi-tour → front stride → final LLL), and `reduce`/`reduce_rows` (FastInt64 scheduler: beta<40 → deep-LLL-4, beta≥40 → progressive schedule). The LLL loop already in `lll_reduce` is extracted into `_lll_tour` for reuse. `api._reduce_basis_reference` gains the BKZ parameters.

**Tech Stack:** Python 3 (`unittest`, `fractions.Fraction`), exact `_exact`/`_bkz`/`verify_sage` modules (unchanged), optional Sage GSO backend.

**Spec:** `docs/spec/bkz-tour-transcription.md` (binding authority; the plan argues from it).

## Global Constraints

- Reference reducer must NOT call Sage `LLL()`/`BKZ()`/fplll as an engine (Sage/flatter are comparison subjects only).
- No floating point in the reference/verifier on the exact path (only `int`/`Fraction`; `_svp`'s `log_volume` float-logging for the GH radius is pre-existing and unchanged).
- Arbitrary-precision arithmetic on the exact path.
- `unittest`, not pytest.
- TDD: every production change is preceded by a failing test that is watched to fail.
- Never `git add t.py`.
- Commit messages end with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- BKZ block size `beta` is bounded by `[2, 80]`.

## Review Focus

The five input classes most likely to bite a user, not covered by the happy-path tests:

1. `beta > n` on a **segment** (HKZ tail) and `beta > n` on the **whole basis** (degenerate HKZ tour) — the strict `>` vs `>=` in `_bkz_segment` and `_advance_front`. → Task 3 test `test_bkz_segment_hkz_tail_verifies` + Task 4 `test_hkz_tour_verifies`.
2. `bkz_size < beta` must raise, not silently under-segment. → Task 3 test `test_bkz_size_below_beta_rejected`.
3. `depth` combined with `beta` must raise. → Task 5 test `test_reduce_rejects_depth_with_beta`.
4. `bkz_prog > beta` must raise (invalid schedule). → Task 5 test `test_reduce_rejects_bad_bkz_prog`.
5. `cur_front % beta` offset when `beta` does not divide `bkz_size` — the segment partition must start mid-rank and still cover the tail. → Task 4 test `test_bkz_multi_segment_multi_tour_verifies`.

---

### Task 1: `_advance_front` transcription

**Files:**
- Modify: `src/blaster/reference.py` (add `_advance_front` after `_progressive_betas`, ~line 284)
- Test: `tests/test_reference.py` (`TestFrontTourSchedule`, lines 770–811)

**Interfaces:**
- Consumes: none (pure function).
- Produces: `_advance_front(cur_front, beta, n, bkz_size) -> (int, int)`, consumed by `bkz_reduce` in Task 4.

- [ ] **Step 1: Write the failing test**

In `tests/test_reference.py`, change `TestFrontTourSchedule.setUp` (line 778–780) to import the reference copy, and add a parity test:

```python
    def setUp(self):
        from blaster.reference import _advance_front
        self._advance_front = _advance_front

    # ... existing five tests (test_stride_*, test_exact_end_*, test_strict_overrun_*,
    #     test_unit_stride_*, test_full_tour_*) are unchanged ...

    def test_matches_fastint64_source_of_truth(self):
        from blaster.blaster import _advance_front as _fast
        for beta in range(2, 9):
            for n in range(beta, 13):
                for bkz_size in range(beta, n + 1):
                    for cur_front in range(0, n):
                        self.assertEqual(
                            self._advance_front(cur_front, beta, n, bkz_size),
                            _fast(cur_front, beta, n, bkz_size),
                            (cur_front, beta, n, bkz_size),
                        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_reference.TestFrontTourSchedule -v`
Expected: FAIL — `ImportError: cannot import name '_advance_front' from 'blaster.reference'`.

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/reference.py`, after `_progressive_betas` (line 284), add:

```python
def _advance_front(cur_front, beta, n, bkz_size):
    """Advance BKZ's reduction front; return ``(new_front, tours_done_delta)``.

    Exact transcription of FastInt64 ``_advance_front``: the front jumps by the
    segment stride ``bkz_size - beta + 1``. Only a *strict* overrun
    (``cur_front + beta > n``) ends a tour (front resets to 0, delta 1); an
    exact fit ``== n`` still advances.
    """
    if cur_front + beta > n:
        return 0, 1
    return cur_front + (bkz_size - beta + 1), 0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_reference.TestFrontTourSchedule -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/reference.py tests/test_reference.py
git commit -m "feat(reference): transcribe _advance_front for the BKZ tour driver

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Extract `_lll_tour` (behaviour-preserving refactor)

**Files:**
- Modify: `src/blaster/reference.py` (extract the in-place loop from `lll_reduce`)

**Interfaces:**
- Consumes: `_local_lll`, `_deeplll_local`, `_size_panel`, `_seysen_panel`, `weak_lll_holds` (all existing).
- Produces: `_lll_tour(cols, Ucols, n, block, delta, depth, use_seysen, tr, gso)` (in-place, no return), consumed by `lll_reduce` (unchanged) and `bkz_reduce` (Tasks 4–5).

- [ ] **Step 1: Extract the helper (no new behaviour)**

This is a pure refactor; the existing `lll_reduce` tests are the gate. In
`src/blaster/reference.py`, insert after `_depth_param` (line 262):

```python
def _lll_tour(cols, Ucols, n, block, delta, depth, use_seysen, tr, gso):
    """In-place segmented-LLL loop shared by ``lll_reduce`` and the BKZ driver.

    Half-offset toggle, per-segment local/deep LLL, then the Seysen/size panel,
    repeating until the whole basis is weakly-LLL-reduced.
    """
    offset = 0
    while True:
        offset = (block // 2) if offset == 0 else 0
        tr.append(("offset", offset, False))
        for i in range(offset, n, block):
            w = min(i + block, n) - i
            tr.append(("segment", i, w))
            if depth:
                _deeplll_local(cols, Ucols, i, min(i + block, n), delta, depth, tr, gso)
            else:
                _local_lll(cols, Ucols, i, min(i + block, n), delta, tr, gso)
        (_seysen_panel if use_seysen else _size_panel)(cols, Ucols, tr, gso)
        mu, r = gso(cols)
        if weak_lll_holds(mu, r, delta):
            tr.append(("offset", offset, True))
            break
```

Replace the body of `lll_reduce` from `offset = 0` (line 326) through the
`break` (line 344) with a single call:

```python
    _lll_tour(cols, Ucols, n, block, d, depth, use_seysen, tr, gso_fn)

    return cols, transpose_cols(Ucols), tr
```

- [ ] **Step 2: Run the full suite to verify no behaviour changed**

Run: `python -m unittest discover -s tests -v`
Expected: all pass (existing `lll_reduce` tests pin the identical output).

- [ ] **Step 3: Commit**

```bash
git add src/blaster/reference.py
git commit -m "refactor(reference): extract _lll_tour from lll_reduce for BKZ reuse

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: `_bkz_segment` kernel + `bkz_reduce` refactored onto it

**Files:**
- Modify: `src/blaster/reference.py` (add `_bkz_segment` after `_svp`, line 399; refactor `bkz_reduce` to call it)
- Test: `tests/test_bkz.py` (add `TestBKZSegment`)

**Interfaces:**
- Consumes: `_svp`, `_deeplll_local` (existing).
- Produces: `_bkz_segment(cols, Ucols, i, w, beta, delta, tr, gso)` (in-place), consumed by the tour driver in Task 4.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bkz.py` (imports already include `_deeplll_local`, `bkz_reduce`; add `_bkz_segment` to the `from blaster.reference import` line at 13):

```python
class TestBKZSegment(unittest.TestCase):
    def _columns(self, rows):
        n = len(rows[0])
        d = delta_to_fraction(0.99)
        cols = [column(to_int_matrix(rows), j) for j in range(n)]
        Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
        return cols, Ucols, [], d

    def test_bkz_segment_hkz_tail_verifies(self):
        # beta > segment width forces an HKZ sweep inside the segment.
        B0 = [[2, 3, 5, 7, 11, 13], [0, 4, 6, 8, 10, 12], [0, 0, 7, 9, 12, 15],
              [0, 0, 0, 11, 13, 17], [0, 0, 0, 0, 17, 19], [0, 0, 0, 0, 0, 23]]
        cols, Ucols, tr, d = self._columns(B0)
        _bkz_segment(cols, Ucols, 2, 3, 4, d, tr, resolve_gso(None))  # segment [2,5), beta 4 > 3
        rep = verify_reduction(B0, transpose_cols(cols), transpose_cols(Ucols), mode="bkz")
        self.assertTrue(rep.verified)

    def test_bkz_size_below_beta_rejected(self):
        from blaster.reference import _bkz_size_param
        for bad in (3, 1, True, 2.5):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "bkz_size"):
                    _bkz_size_param(bad, 4, 6)

    def test_bkz_size_capped_at_rank(self):
        from blaster.reference import _bkz_size_param
        self.assertEqual(_bkz_size_param(10, 4, 6), 6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_bkz.TestBKZSegment -v`
Expected: FAIL — `ImportError: cannot import name '_bkz_segment'` (and
`_bkz_size_param`) from `blaster.reference`.

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/reference.py`, after `_svp` (line 399), add:

```python
def _bkz_segment(cols, Ucols, i, w, beta, delta, tr, gso):
    """Exact transcription of the C++ inner ``bkz_reduce`` on one ``w``-wide segment.

    Deep-LLL-4 the segment ``[i, i+w)``, then run the SVP sweep: an HKZ tour
    (``svp(j, i+w-j)``) when the block is wider than the segment, else a flat
    ``svp(j, beta)`` sweep.
    """
    _deeplll_local(cols, Ucols, i, i + w, delta, 4, tr, gso)
    if beta > w:
        for j in range(i, i + w - 1):
            _svp(cols, Ucols, j, i + w - j, delta, tr, gso)
    else:
        for j in range(i, i + w - beta + 1):
            _svp(cols, Ucols, j, beta, delta, tr, gso)


def _bkz_size_param(bkz_size, beta, n):
    """Validate ``bkz_size``: an integer at least ``beta``, capped to the rank."""
    if isinstance(bkz_size, bool) or not isinstance(bkz_size, Integral):
        raise ValueError("bkz_size must be an integer")
    bkz_size = int(bkz_size)
    if bkz_size < beta:
        raise ValueError("bkz_size must be at least beta")
    return min(bkz_size, n)
```

Refactor the existing `bkz_reduce` (lines 411–440) to route through `_bkz_segment`
while keeping its single-sweep, `beta <= 2`-short-circuit semantics:

```python
def bkz_reduce(B0, delta=0.99, *, beta, gso=None):
    """Exact single-sweep BKZ returning ``(cols, U_rows, transcript)``.

    Kept as the whole-basis segment kernel: deep-LLL-4 pre-pass, then one
    ``_bkz_segment(0, n, beta)`` sweep. ``beta <= 2`` returns after the pre-pass.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    if not full_column_rank(B0_rows):
        raise ValueError("basis vectors must be linearly independent")
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    beta = _beta_param(beta)
    gso_fn = resolve_gso(gso)
    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []
    if beta <= 2:
        _deeplll_local(cols, Ucols, 0, n, d, 4, tr, gso_fn)
        return cols, transpose_cols(Ucols), tr
    _bkz_segment(cols, Ucols, 0, n, beta, d, tr, gso_fn)
    return cols, transpose_cols(Ucols), tr
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_bkz.TestBKZSegment tests.test_bkz.TestBKZReduce -v`
Expected: PASS (new 3 + existing 4); `test_bkz2_is_deeplll4`, `test_bkz_tour_verifies`, `test_hkz_tour_verifies`, `test_bad_beta_rejected` all still green.

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/reference.py tests/test_bkz.py
git commit -m "feat(reference): extract _bkz_segment kernel from bkz_reduce

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: `bkz_reduce` → the multi-tour driver

**Files:**
- Modify: `src/blaster/reference.py` (rewrite `bkz_reduce` signature + body)
- Test: `tests/test_bkz.py` (migrate `test_bkz2_is_deeplll4`; add multi-tour tests)

**Interfaces:**
- Consumes: `_bkz_segment`, `_advance_front`, `_lll_tour`, `_bkz_size_param`, `_bkz_tours_param` (add now), `_segment_width`.
- Produces: `bkz_reduce(B0, delta=0.99, *, beta, gso=None, lll_size=None, use_seysen=False, bkz_tours=8, bkz_size=None)`, consumed by `reduce()` in Task 5 and `api` in Task 6.

- [ ] **Step 1: Write the failing test**

In `tests/test_bkz.py`, replace `test_bkz2_is_deeplll4` (lines 94–100) with a
direct-call beta=2 verification, and add a multi-tour test to `TestBKZReduce`:

```python
    def test_bkz2_tour_verifies(self):
        # Direct beta=2 call now runs a full (degenerate) tour and still verifies.
        B0 = [[2, 3, 5], [0, 4, 6], [0, 0, 7]]
        cols, U_rows, tr = bkz_reduce(B0, beta=2)
        rep = verify_reduction(B0, transpose_cols(cols), U_rows, mode="bkz")
        self.assertTrue(rep.verified)

    def test_bkz_multi_segment_multi_tour_verifies(self):
        # n=6, beta=3, bkz_size=4 -> front stride 2; 2 tours visit several fronts.
        B0 = [[2, 3, 5, 7, 11, 13], [0, 4, 6, 8, 10, 12], [0, 0, 7, 9, 12, 15],
              [0, 0, 0, 11, 13, 17], [0, 0, 0, 0, 17, 19], [0, 0, 0, 0, 0, 23]]
        cols, U_rows, tr = bkz_reduce(B0, beta=3, bkz_size=4, bkz_tours=2)
        rep = verify_reduction(B0, transpose_cols(cols), U_rows, mode="bkz")
        self.assertTrue(rep.verified)
```

`test_bkz_tour_verifies`, `test_hkz_tour_verifies`, `test_bad_beta_rejected`
are unchanged (they keep passing under the new driver).

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_bkz.TestBKZReduce.test_bkz2_tour_verifies tests.test_bkz.TestBKZReduce.test_bkz_multi_segment_multi_tour_verifies -v`
Expected: FAIL — `bkz_reduce` has no `bkz_size`/`bkz_tours`/`lll_size` parameters (`TypeError`), and beta=2 is short-circuited so the tour path is not exercised.

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/reference.py`, replace `bkz_reduce` (the Task 3 version) with
the tour driver, and add `_bkz_tours_param` near `_bkz_size_param`:

```python
def _bkz_tours_param(bkz_tours):
    if isinstance(bkz_tours, bool) or not isinstance(bkz_tours, Integral):
        raise ValueError("bkz_tours must be an integer")
    bkz_tours = int(bkz_tours)
    if bkz_tours < 1:
        raise ValueError("bkz_tours must be at least 1")
    return bkz_tours


def bkz_reduce(B0, delta=0.99, *, beta, gso=None, lll_size=None,
               use_seysen=False, bkz_tours=8, bkz_size=None):
    """Exact progressive/segmented BKZ tour returning ``(cols, U_rows, transcript)``.

    Transcribes FastInt64's Python ``bkz_reduce``: pre-reduce the whole basis
    with deep-LLL-4 (full iterative LLL), then run ``bkz_tours`` tours. Each tour
    partitions ``[cur_front % beta, n)`` into ``bkz_size``-wide segments, reduces
    each with :func:`_bkz_segment`, applies the Seysen/size panel, advances the
    front with :func:`_advance_front`, and finishes with another full LLL pass.
    ``beta > n`` degenerates to a repeated whole-basis HKZ tour. Never calls Sage.
    """
    B0_rows = to_int_matrix(B0, name="B0")
    if not full_column_rank(B0_rows):
        raise ValueError("basis vectors must be linearly independent")
    n = len(B0_rows[0])
    d = delta_to_fraction(delta)
    beta = _beta_param(beta)
    gso_fn = resolve_gso(gso)
    block = _segment_width(lll_size, n)
    bkz_tours = _bkz_tours_param(bkz_tours)
    if bkz_size is None:
        bkz_size = max(block, beta)
    bkz_size = _bkz_size_param(bkz_size, beta, n)

    cols = [column(B0_rows, j) for j in range(n)]
    Ucols = [[1 if r == j else 0 for r in range(n)] for j in range(n)]
    tr = []

    _lll_tour(cols, Ucols, n, block, d, 4, use_seysen, tr, gso_fn)
    tours_done, cur_front = 0, 0
    while tours_done < bkz_tours:
        for i in range(cur_front % beta, n, bkz_size):
            w = min(n - i, bkz_size)
            tr.append(("segment", i, w))
            _bkz_segment(cols, Ucols, i, w, beta, d, tr, gso_fn)
        (_seysen_panel if use_seysen else _size_panel)(cols, Ucols, tr, gso_fn)
        cur_front, tours_done_delta = _advance_front(cur_front, beta, n, bkz_size)
        tours_done += tours_done_delta
        _lll_tour(cols, Ucols, n, block, d, 4, use_seysen, tr, gso_fn)
    return cols, transpose_cols(Ucols), tr
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_bkz.TestBKZReduce tests.test_bkz.TestBKZSegment -v`
Expected: PASS (TestBKZReduce 5 tests, TestBKZSegment 3 tests).

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/reference.py tests/test_bkz.py
git commit -m "feat(reference): rewrite bkz_reduce as the segmented multi-tour driver

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: `reduce` / `reduce_rows` scheduler (wire `_progressive_betas`)

**Files:**
- Modify: `src/blaster/reference.py` (add `_bkz_prog_param`, `reduce`, `reduce_rows` after `bkz_reduce`)
- Test: `tests/test_bkz.py` (add `TestReferenceReduce`)

**Interfaces:**
- Consumes: `bkz_reduce`, `lll_reduce`, `lll_reduce_rows`, `_progressive_betas`, `_bkz_size_param`, `_bkz_tours_param`, `_segment_width`.
- Produces: `reduce(B0, delta=0.99, *, gso=None, lll_size=None, use_seysen=False, depth=0, beta=None, bkz_tours=8, bkz_size=None, bkz_prog=None)` and `reduce_rows(...)`, consumed by `api` in Task 6.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bkz.py` (add `reduce, reduce_rows` to the `from blaster.reference import` line at 13):

```python
class TestReferenceReduce(unittest.TestCase):
    def test_reduce_beta_none_is_lll(self):
        B0 = [[2, 3, 5, 7], [0, 4, 6, 8], [0, 0, 7, 9], [0, 0, 0, 11]]
        cols_a, U_a, tr_a = reduce(B0)
        cols_b, U_b, tr_b = lll_reduce(B0)
        self.assertEqual(cols_a, cols_b)
        self.assertEqual(U_a, U_b)
        self.assertEqual(tr_a, tr_b)

    def test_reduce_beta_below_40_is_deeplll4(self):
        B0 = [[2, 3, 5, 7], [0, 4, 6, 8], [0, 0, 7, 9], [0, 0, 0, 11]]
        cols_a, U_a, tr_a = reduce(B0, beta=2)
        cols_b, U_b, tr_b = lll_reduce(B0, depth=4)
        self.assertEqual(cols_a, cols_b)
        self.assertEqual(U_a, U_b)
        self.assertEqual(tr_a, tr_b)

    def test_reduce_rejects_depth_with_beta(self):
        B0 = [[2, 3], [0, 4]]
        with self.assertRaisesRegex(ValueError, "depth"):
            reduce(B0, beta=4, depth=2)

    def test_reduce_rejects_beta_above_rank(self):
        B0 = [[2, 3], [0, 4]]
        with self.assertRaisesRegex(ValueError, "rank"):
            reduce(B0, beta=4)

    def test_reduce_rejects_bad_bkz_prog(self):
        B0 = [[2, 3], [0, 4]]
        with self.assertRaisesRegex(ValueError, "bkz_prog"):
            reduce(B0, beta=2, bkz_prog=5)

    def test_reduce_follows_progressive_schedule(self):
        from unittest import mock
        import blaster.reference as reference
        B0 = [[2] + [0] * 41] + [[0] * i + [2] + [0] * (41 - i) for i in range(1, 42)]
        calls = []
        def spy(B0, delta=0.99, *, beta, gso=None, lll_size=None, use_seysen=False,
                bkz_tours=8, bkz_size=None):
            calls.append((beta, bkz_tours))
            return ([], [], [])
        with mock.patch.object(reference, "bkz_reduce", side_effect=spy):
            reference.reduce(B0, beta=42, bkz_prog=1)
        self.assertEqual(calls, [(40, 1), (41, 1), (42, 8)])

    def test_reduce_rows_round_trips(self):
        rows = [[2, 3, 5, 7], [0, 4, 6, 8], [0, 0, 7, 9], [0, 0, 0, 11]]
        reduced, T, tr = reduce_rows(rows, beta=2)
        from blaster.verify import verify_reduction
        self.assertTrue(verify_reduction(rows, reduced, T, mode="bkz").verified)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_bkz.TestReferenceReduce -v`
Expected: FAIL — `ImportError: cannot import name 'reduce' from 'blaster.reference'`.

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/reference.py`, after `bkz_reduce`, add:

```python
def _bkz_prog_param(bkz_prog, beta):
    if isinstance(bkz_prog, bool) or not isinstance(bkz_prog, Integral):
        raise ValueError("bkz_prog must be an integer")
    bkz_prog = int(bkz_prog)
    if bkz_prog < 1 or bkz_prog > beta:
        raise ValueError("bkz_prog must be in [1, beta]")
    return bkz_prog


def reduce(B0, delta=0.99, *, gso=None, lll_size=None, use_seysen=False,
           depth=0, beta=None, bkz_tours=8, bkz_size=None, bkz_prog=None):
    """Column-convention reducer mirroring FastInt64 ``reduce()``.

    ``beta is None`` selects segmented LLL; ``beta < 40`` selects deep-LLL-4
    preprocessing (no tour); ``beta >= 40`` runs the progressive-beta schedule
    from :func:`_progressive_betas`. ``depth`` cannot be combined with ``beta``.
    Returns ``(cols, U_rows, transcript)``.
    """
    if beta is None:
        return lll_reduce(B0, delta, gso=gso, lll_size=lll_size,
                          use_seysen=use_seysen, depth=depth)
    beta = _beta_param(beta)
    if depth:
        raise ValueError("depth cannot be combined with beta")
    B0_rows = to_int_matrix(B0, name="B0")
    if not full_column_rank(B0_rows):
        raise ValueError("basis vectors must be linearly independent")
    n = len(B0_rows[0])
    if beta > n:
        raise ValueError("beta must not exceed the basis rank")
    block = _segment_width(lll_size, n)
    bkz_tours = _bkz_tours_param(bkz_tours)
    if bkz_size is None:
        bkz_size = max(block, beta)
    bkz_size = _bkz_size_param(bkz_size, beta, n)
    if bkz_prog is None:
        bkz_prog = beta
    bkz_prog = _bkz_prog_param(bkz_prog, beta)

    if beta < 40:
        return lll_reduce(B0, delta, gso=gso, lll_size=lll_size,
                          use_seysen=use_seysen, depth=4)

    cols, U_rows, tr = None, None, None
    for beta_ in _progressive_betas(beta, bkz_prog):
        cols, U_rows, tr = bkz_reduce(
            B0, delta, beta=beta_, gso=gso, lll_size=lll_size,
            use_seysen=use_seysen,
            bkz_tours=bkz_tours if beta_ == beta else 1,
            bkz_size=bkz_size,
        )
    return cols, U_rows, tr


def reduce_rows(A0, delta=0.99, *, gso=None, lll_size=None, use_seysen=False,
                depth=0, beta=None, bkz_tours=8, bkz_size=None, bkz_prog=None):
    """Row-convention reducer: ``A0`` rows are the basis vectors.

    Returns ``(rows, T, transcript)`` with ``T @ A0 == rows`` — the transpose of
    :func:`reduce`.
    """
    A0_rows = to_int_matrix(A0, name="A0")
    cols, U_rows, tr = reduce(
        transpose_cols(A0_rows), delta, gso=gso, lll_size=lll_size,
        use_seysen=use_seysen, depth=depth, beta=beta, bkz_tours=bkz_tours,
        bkz_size=bkz_size, bkz_prog=bkz_prog,
    )
    rows = [c[:] for c in cols]
    T = transpose_cols(U_rows)
    return rows, T, tr
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_bkz.TestReferenceReduce -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/reference.py tests/test_bkz.py
git commit -m "feat(reference): add reduce/reduce_rows scheduler wiring _progressive_betas

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: Public API wiring (`reduce_basis(engine="reference", beta=...)`)

**Files:**
- Modify: `src/blaster/api.py` (import `reduce`, `reduce_rows`; extend `_reduce_basis_reference`)
- Test: `tests/test_api.py` (add `ReferenceBKZApiTests`)

**Interfaces:**
- Consumes: `reduce`, `reduce_rows` from `blaster.reference` (Task 5).
- Produces: `reduce_basis(engine="reference", beta=..., bkz_tours=..., bkz_size=..., bkz_prog=...)` end-to-end.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:

```python
class ReferenceBKZApiTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array(
            [[2, 3, 5, 7], [0, 4, 6, 8], [0, 0, 7, 9], [0, 0, 0, 11]], dtype=object
        )

    def test_reference_bkz_returns_reduced(self):
        reduced = reduce_basis(self.rows, engine="reference", beta=2)
        from blaster.verify import verify_reduction
        self.assertTrue(verify_reduction(self.rows.tolist(), reduced.tolist(),
                                         np.eye(4, dtype=object).tolist(), mode="bkz").structural_ok)

    def test_reference_bkz_return_transform(self):
        reduced, transform, transcript = reduce_basis(
            self.rows, engine="reference", beta=2, return_transform=True
        )
        self.assertIsInstance(transcript, list)

    def test_reference_bkz_rejects_depth(self):
        with self.assertRaisesRegex(ValueError, "depth"):
            reduce_basis(self.rows, engine="reference", beta=2, depth=2)

    def test_reference_still_rejects_unknown_params(self):
        with self.assertRaisesRegex(ValueError, "cores"):
            reduce_basis(self.rows, engine="reference", cores=1)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_api.ReferenceBKZApiTests -v`
Expected: FAIL — `reduce_basis(engine="reference", beta=2)` raises
`ValueError` because `beta` is not in the reference allow-list (existing
`_reduce_basis_reference` rejects it).

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/api.py`, change line 8 and `_reduce_basis_reference`.

> **Why aliases:** `api.py:7` already does `from .blaster import reduce` for the
> fast path (`reduce_basis` calls it at line 59). Importing the reference
> `reduce` under the same name would shadow it and break `engine="fast"`. The
> reference functions are therefore imported under aliases.

```python
from .reference import (
    lll_reduce,
    lll_reduce_rows,
    reduce as reference_reduce,
    reduce_rows as reference_reduce_rows,
)
```

```python
def _reduce_basis_reference(basis, *, orientation, return_transform, **parameters):
    """Exact arbitrary-precision LLL/BKZ path of :func:`reduce_basis`."""
    delta = parameters.pop("delta", 0.99)
    gso = parameters.pop("gso", None)
    lll_size = parameters.pop("lll_size", None)
    use_seysen = parameters.pop("use_seysen", False)
    depth = parameters.pop("depth", 0)
    beta = parameters.pop("beta", None)
    bkz_tours = parameters.pop("bkz_tours", 8)
    bkz_size = parameters.pop("bkz_size", None)
    bkz_prog = parameters.pop("bkz_prog", None)
    if parameters:
        raise ValueError(
            "reference engine supports only 'delta', 'gso', 'lll_size', "
            "'use_seysen', 'depth', 'beta', 'bkz_tours', 'bkz_size' and "
            "'bkz_prog', got: " + ", ".join(sorted(parameters))
        )

    sage_input = is_sage_matrix(basis)
    sage_ring = basis.base_ring() if sage_input else None
    rows = to_int_matrix(basis, name="basis")

    common = dict(delta=delta, gso=gso, lll_size=lll_size, use_seysen=use_seysen)
    if beta is not None:
        if orientation == "rows":
            reduced_rows, transform, transcript = reference_reduce_rows(
                rows, **common, depth=depth, beta=beta,
                bkz_tours=bkz_tours, bkz_size=bkz_size, bkz_prog=bkz_prog)
        else:
            cols, U_rows, transcript = reference_reduce(
                rows, **common, depth=depth, beta=beta,
                bkz_tours=bkz_tours, bkz_size=bkz_size, bkz_prog=bkz_prog)
            reduced_rows = transpose_cols(cols)
            transform = U_rows
    elif orientation == "rows":
        reduced_rows, transform, transcript = lll_reduce_rows(
            rows, **common, depth=depth)
    else:
        cols, U_rows, transcript = lll_reduce(rows, **common, depth=depth)
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
```

Also extend the `engine="reference"` paragraph of `reduce_basis`'s docstring to
mention `beta`, `bkz_tours`, `bkz_size`, `bkz_prog`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_api.ReferenceBKZApiTests -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/api.py tests/test_api.py
git commit -m "feat(reference): expose BKZ parameters through reduce_basis(engine='reference')

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: Multicore deferred minors (C-7 polish)

**Files:**
- Modify: `src/blaster/blaster.py` (`reduce()` docstring; `_resolve_cores` annotation)
- Modify: `tests/test_api.py` (`MultiCoreSwitchTests`: cores=2 e2e + multi-segment determinism)
- Modify: `tests/test_io_cli.py` (`CommandLineTests`: CLI auto asserts the reduced result, not just shape)

**Interfaces:**
- Consumes: `reduce`, `_resolve_cores`, `build_parser` (existing).
- Produces: no new public surface — tightened tests + two annotations.

- [ ] **Step 1: Write the failing tests**

In `tests/test_api.py::MultiCoreSwitchTests`, add:

```python
    def test_reduce_cores_two_matches_single_threaded(self):
        t1, r1, _ = reduce(self.columns, lll_size=3, cores=1, use_seysen=False)
        t2, r2, _ = reduce(self.columns, lll_size=3, cores=2, use_seysen=False)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(t1, t2)

    def test_reduce_multi_segment_deterministic_across_cores(self):
        # rank-3 with lll_size=3 is a single segment; use a wider basis so the
        # Cython prange actually splits into multiple segments.
        wide = np.array(
            [[8, 1, 0, 0, 0, 0, 0, 0],
             [0, 8, 1, 0, 0, 0, 0, 0],
             [0, 0, 8, 1, 0, 0, 0, 0],
             [0, 0, 0, 8, 1, 0, 0, 0],
             [0, 0, 0, 0, 8, 1, 0, 0],
             [0, 0, 0, 0, 0, 8, 1, 0],
             [0, 0, 0, 0, 0, 0, 8, 1],
             [0, 0, 0, 0, 0, 0, 0, 8]], dtype=np.int64
        )
        _, r1, _ = reduce(wide, lll_size=4, cores=1, use_seysen=False)
        _, ra, _ = reduce(wide, lll_size=4, cores="auto", use_seysen=False)
        np.testing.assert_array_equal(r1, ra)
```

In `tests/test_io_cli.py`, strengthen `test_cli_cores_auto_reduces` (line 211–219)
to assert the reduced lattice has strictly decreased profile, and add:

```python
    def test_cli_cores_integer_matches_auto(self):
        source = io.StringIO()
        write_lattice(self.columns, source)
        auto = self.run_cli("--no-seysen", "--cores", "auto", input_text=source.getvalue())
        source.seek(0)
        one = self.run_cli("--no-seysen", "--cores", "1", input_text=source.getvalue())
        self.assertEqual(auto.returncode, 0, auto.stderr)
        self.assertEqual(one.returncode, 0, one.stderr)
        self.assertEqual(auto.stdout, one.stdout)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_api.MultiCoreSwitchTests.test_reduce_cores_two_matches_single_threaded tests.test_api.MultiCoreSwitchTests.test_reduce_multi_segment_deterministic_across_cores tests.test_io_cli.CommandLineTests.test_cli_cores_integer_matches_auto -v`
Expected: the two `test_api` cases already pass at the current defaults (they
pin `cores=2` / multi-segment determinism that M-2's tests did not); `test_cli_cores_integer_matches_auto` PASS or FAIL depending on whether `auto` and `1`
produce byte-identical output — if PASS, keep it as a regression pin (no code
change needed); if FAIL, note it in the commit as a genuine output divergence to
investigate, and fix per Task 7 Step 3.

- [ ] **Step 3: Write the minimal implementation**

3a. In `src/blaster/blaster.py`, annotate `_resolve_cores` (line 185):

```python
def _resolve_cores(cores) -> int:
```

3b. Extend the `reduce()` docstring's `cores` sentence (near line 307) with:
"``cores`` may be ``\"auto\"`` (default; all logical cores) or an integer ``>= 1``."

No behavioural change to the library; the CLI already passes `"auto"`/`int` through.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_api.MultiCoreSwitchTests tests.test_io_cli.CommandLineTests -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/blaster.py tests/test_api.py tests/test_io_cli.py
git commit -m "test(M-2): pin cores=2 and multi-segment determinism; annotate _resolve_cores

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** §3.1 → Task 1; §3.2 → Task 3; §3.3 → Task 4; §3.4 → Task 5; §3.5 → Task 2; §2.5 (public API) → Task 6; §7 C-7 → Task 7. B/D/C-8/C-9 are spec §7 non-goals (no tasks).
- **Placeholder scan:** none — every step has concrete code and a concrete command.
- **Type consistency:** `_bkz_size_param` / `_bkz_tours_param` are defined (Task 3/4) before `reduce()` consumes them (Task 5); `_advance_front` (Task 1) and `_lll_tour` (Task 2) precede `bkz_reduce` (Task 4); `reduce`/`reduce_rows` (Task 5) precede `api` (Task 6). Signatures match across tasks.
- **Review Focus:** all five lines have a named test in the owning task (Task 3/4/5).
