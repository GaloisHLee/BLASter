# BKZ full-tour transcription + progressive-schedule wiring

Status: accepted design (M-3). Version 0.1. Date 2026-10-15.

This document defines the last precision-track deliverable for the reference
(exact, arbitrary-precision) engine: completing its BKZ coverage from the
"per-segment kernel" to the **full FastInt64 progressive/segmented BKZ tour**,
and wiring the already-transcribed `_progressive_betas` schedule into the
reference reducer's runtime. It is the spec that M-3 is implemented against.
It is paired with the "not-done" audit (A/B/C/D) from which it was derived; the
B and D items are recorded here as deliberate non-goals so they are not
re-read as debt.

## 1. Background: the three BKZ layers

FastInt64's BKZ is three layers. The reference engine transcribes each one
separately; the gap this spec closes is the *middle* layer.

| Layer | FastInt64 location | Reference transcription | Status |
|---|---|---|---|
| Scheduler | `blaster.py:reduce()` BKZ branch (lines 389–403) | `reference.reduce()` / `reduce_rows()` | **NEW** (this spec) |
| Tour driver | `blaster.py:bkz_reduce()` (lines 120–173) | `reference.bkz_reduce()` | **REWRITTEN** (this spec) |
| Segment kernel | C++ `bkz_reduce(N,R,U,delta,beta)` (`core/lattice_reduction.cpp:309`), invoked per segment by `block_bkz` (`core/blaster.pyx:111`) | `reference._bkz_segment()` | **NEW** (generalizes today's `bkz_reduce`) |
| Front stride | `blaster.py:_advance_front()` (lines 104–117) | `reference._advance_front()` | **NEW** |
| Beta schedule | `blaster.py:_progressive_betas()` (lines 291–298) | `reference._progressive_betas()` | done, unwired (D-1) |

What exists today: `reference.bkz_reduce` is a faithful transcription of the
**C++ inner kernel** — a single `_deeplll_local(0, n, 4)` pre-pass followed by a
flat `svp(i, beta)` sweep (or an HKZ `svp(i, n-i)` sweep when `beta > n`), with a
`beta <= 2` short-circuit. It has no notion of tours, fronts, or `bkz_size`
segmentation. This spec promotes that body into `_bkz_segment` (one segment)
and rewrites `bkz_reduce` as the multi-tour driver.

## 2. What this builds

1. **`_advance_front(cur_front, beta, n, bkz_size) -> (int, int)`** — pure
   transcription of `blaster.py:_advance_front`. Strict `>` is the wrap gate:
   `cur_front + beta > n` → `(0, 1)`; otherwise `(cur_front + (bkz_size - beta + 1), 0)`.

2. **`_bkz_segment(cols, Ucols, i, w, beta, delta, tr, gso)`** — the C++ inner
   kernel generalized to a `w`-wide segment at column offset `i`: deep-LLL-4
   over `[i, i+w)`, then an HKZ sweep (`svp(j, i+w-j)` for `j` in `[i, i+w-1)`)
   when `beta > w`, else a flat sweep (`svp(j, beta)` for `j` in `[i, i+w-beta+1)`).

3. **`bkz_reduce(...)` — the tour driver.** New signature
   `bkz_reduce(B0, delta=0.99, *, beta, gso=None, lll_size=None, use_seysen=False, bkz_tours=8, bkz_size=None)`,
   transcribing `blaster.py:bkz_reduce`:
   - pre-reduce the whole basis with one full iterative LLL pass at `depth=4`
     (the extracted `_lll_tour`),
   - `while tours_done < bkz_tours`: partition `[cur_front % beta, n)` into
     `bkz_size`-wide segments and run `_bkz_segment` on each; apply the
     Seysen/size panel; advance the front with `_advance_front`; finish with a
     full LLL pass.

4. **`reduce()` / `reduce_rows()` — the scheduler**, transcribing
   `blaster.py:reduce()`'s three-way BKZ branch and wiring `_progressive_betas`:
   - `beta is None` → `lll_reduce(...)` (unchanged LLL);
   - `beta < 40` → `lll_reduce(..., depth=4)` (deep-LLL-4 preprocessing, no tour);
   - `beta >= 40` → `for beta_ in _progressive_betas(beta, bkz_prog): bkz_reduce(..., beta_, bkz_tours if beta_ == beta else 1, bkz_size)`.

5. **Public API wiring.** `api._reduce_basis_reference` routes `beta` (with
   `bkz_tours`/`bkz_size`/`bkz_prog`) to `reduce()`/`reduce_rows()`, so
   `reduce_basis(engine="reference", beta=...)` is exposed end-to-end.

## 3. Exact contract

### 3.1 `_advance_front(cur_front, beta, n, bkz_size)`

```
if cur_front + beta > n: return (0, 1)
return (cur_front + (bkz_size - beta + 1), 0)
```

Strict `>` only. An exact fit `cur_front + beta == n` advances (does not wrap).

### 3.2 `_bkz_segment(cols, Ucols, i, w, beta, delta, tr, gso)`

```
_deeplll_local(cols, Ucols, i, i + w, delta, 4, tr, gso)
if beta > w:
    for j in range(i, i + w - 1):  _svp(cols, Ucols, j, i + w - j, delta, tr, gso)
else:
    for j in range(i, i + w - beta + 1):  _svp(cols, Ucols, j, beta, delta, tr, gso)
```

`_svp` is unchanged: its post-step already re-reduces `[0, min(j+w, n))` with
depth 4, exactly as the C++ `svp` does (global indices, not segment-relative).

### 3.3 `bkz_reduce` (tour driver)

Validation: `_beta_param(beta)` (int in `[2, 80]`); `_segment_width(lll_size, n)`
for the LLL pre/post pass; `_bkz_tours_param(bkz_tours)` (int `>= 1`);
`_bkz_size_param(bkz_size, beta, n)` (int `>= beta`, capped to `n`; default
`max(block, beta)`). `beta > n` is permitted and degenerates to a repeated
whole-basis HKZ tour (preserving the existing HKZ test).

Body (column convention, `B0 * U == B`, returns `(cols, U_rows, tr)`):

```
cols, Ucols, tr = setup(B0)             # identity U, empty transcript
_lll_tour(cols, Ucols, n, block, d, 4, use_seysen, tr, gso)   # pre-reduce, depth 4
tours_done, cur_front = 0, 0
while tours_done < bkz_tours:
    for i in range(cur_front % beta, n, bkz_size):
        w = min(n - i, bkz_size)
        tr.append(("segment", i, w))
        _bkz_segment(cols, Ucols, i, w, beta, d, tr, gso)
    (_seysen_panel if use_seysen else _size_panel)(cols, Ucols, tr, gso)
    cur_front, dd = _advance_front(cur_front, beta, n, bkz_size)
    tours_done += dd
    _lll_tour(cols, Ucols, n, block, d, 4, use_seysen, tr, gso)
return cols, transpose_cols(Ucols), tr
```

### 3.4 `reduce` / `reduce_rows` (scheduler)

`reduce(B0, delta=0.99, *, gso=None, lll_size=None, use_seysen=False, depth=0, beta=None, bkz_tours=8, bkz_size=None, bkz_prog=None)`:

```
if beta is None: return lll_reduce(...)                       # LLL path
beta = _beta_param(beta)
if depth: raise ValueError("depth cannot be combined with beta")
n = rank(B0)
if beta > n: raise ValueError("beta must not exceed the basis rank")
bkz_tours = _bkz_tours_param(bkz_tours)
bkz_size = _bkz_size_param(max(block, beta) if bkz_size is None else bkz_size, beta, n)
bkz_prog = beta if bkz_prog is None else _bkz_prog_param(bkz_prog, beta)   # int in [1, beta]
if beta < 40: return lll_reduce(..., depth=4)                 # deep-LLL-4, no tour
for beta_ in _progressive_betas(beta, bkz_prog):
    cols, U_rows, tr = bkz_reduce(B0, delta, beta=beta_, gso=gso, lll_size=lll_size,
                                  use_seysen=use_seysen,
                                  bkz_tours=bkz_tours if beta_ == beta else 1,
                                  bkz_size=bkz_size)
return cols, U_rows, tr
```

`reduce_rows(A0, ...)` transposes in, calls `reduce`, transposes back
(`T @ A0 == rows`).

### 3.5 `_lll_tour` extraction

The in-place loop currently inlined in `lll_reduce` (offset toggle → per-segment
`_local_lll`/`_deeplll_local` → panel → `weak_lll_holds` termination) is
extracted verbatim into `_lll_tour(cols, Ucols, n, block, delta, depth, use_seysen, tr, gso)`.
`lll_reduce` becomes setup + one `_lll_tour` call; its behaviour is unchanged
(existing LLL tests stay green). The BKZ driver reuses it for its pre/post passes.

## 4. Semantics and invariants

- **Reference engine stays exact.** No floating point enters the reduction path
  (`_svp`'s `log_volume` is already `float`-logged for the GH radius, matching the
  existing enumeration path; all decisions are `Fraction`/`int`).
- **Never calls Sage `LLL`/`BKZ`/fplll.** `gso` only swaps the exact
  Gram-Schmidt backend (`"rational"` / `"sage"` / callable).
- **`beta <= 2` behaviour moves, not disappears.** The old `bkz_reduce`
  `beta <= 2` short-circuit was the beta<40 branch in disguise; it now lives in
  `reduce()` as `beta < 40 → lll_reduce(depth=4)`.
- **`bkz_reduce` remains callable directly** with any `beta` in `[2, 80]`;
  `beta > n` is a degenerate HKZ tour (preserves `test_hkz_tour_verifies`).
- **Transcript format is unchanged.** Only `("offset", …)`, `("segment", …)`,
  `("add", …)`, `("swap", …)`, `("deep", …)` records are emitted; the tour
  driver adds `("segment", i, w)` records ahead of each `_bkz_segment`, same shape
  as `_lll_tour`'s.

## 5. Worked examples

Rank-5 upper-triangular basis `B0` (columns), `delta=0.99`.

- `bkz_reduce(B0, beta=3)` → pre-reduces with depth-4 LLL, then runs 8 tours of
  `bkz_size = max(5, 3) = 5` → a single whole-basis segment per tour, HKZ only if
  a segment is narrower than 3.
- `bkz_reduce(B0, beta=3, bkz_size=4)` → segments `[0,4)`, `[4,5)`, front stride
  `4-3+1 = 2`.
- `bkz_reduce(B0, beta=4)` (`n=5`) → `bkz_size=5`, beta 4 ≤ n, flat sweep.
- `reduce(B0, beta=2)` → `lll_reduce(B0, depth=4)` (deep-LLL-4, no tour).
- `reduce(B0, beta=40, bkz_prog=8)` → `_progressive_betas(40, 8) == (40,)`.
- `reduce(B0, beta=42, bkz_prog=1)` → schedule `(40, 41, 42)`, tours `(1, 1, 8)`.
- `reduce(B0, beta=4, depth=2)` → `ValueError("depth cannot be combined with beta")`.
- `reduce_basis(rows, engine="reference", beta=3)` → reduced rows (and, with
  `return_transform=True`, the transform + transcript).

## 6. Acceptance driver (drives the tests)

Tests live in `tests/test_reference.py` (`_advance_front`), `tests/test_bkz.py`
(kernel, tour, scheduler), `tests/test_api.py` (public wiring). Each is written
first and observed to fail (TDD).

1. `reference._advance_front` matches `blaster.blaster._advance_front` over
   `beta in [2, 9)`, `n in [beta, 13)`, `bkz_size in [beta, n+1)`,
   `cur_front in [0, n)`, and the five existing F-4 front-tour cases pass against
   the reference copy.
2. `_bkz_segment(0, n, beta)` reproduces the old single-sweep `bkz_reduce`
   (existing `test_bkz_tour_verifies` / `test_hkz_tour_verifies` keep passing);
   a segment at `i > 0` reduces its window and the result verifies.
3. `bkz_reduce(beta=3)` (tour, defaults) verifies; `bkz_reduce(beta=3, bkz_size=4,
   bkz_tours=2)` (multi-segment, multi-tour) verifies; `bkz_reduce(beta=4)` on
   `n=3` (HKZ) verifies; `bkz_reduce(beta=2)` verifies.
4. `_bkz_size_param(bkz_size, beta, n)` raises `ValueError` matching `"bkz_size"`
   for `bkz_size < beta` and for non-int values; `_bkz_tours_param` / `_bkz_prog_param`
   reject bad values.
5. `reduce(beta=None)` equals `lll_reduce`; `reduce(beta=2)` equals
   `lll_reduce(depth=4)`; `reduce(beta=4, depth=2)` raises; `reduce(beta=rank+1)`
   raises `"beta must not exceed the basis rank"`.
6. `reduce(beta=42, bkz_prog=1)` dispatches `bkz_reduce` with
   `[(40, 1), (41, 1), (42, 8)]` (schedule + progressive-tour counts), observed
   by a spy on `blaster.reference.bkz_reduce`; and `reduce(beta=44, bkz_prog=4)`
   produces a verifying result end-to-end.
7. `reduce_basis(engine="reference", beta=3)` (rows) returns a verifying reduced
   matrix; `return_transform=True` returns `(reduced, transform, transcript)`;
   `reduce_basis(engine="reference", beta=3, depth=2)` raises;
   `reduce_basis(engine="reference", cores=1)` still raises (not in the allow-list).

## 7. Out of scope (from the A/B/C/D audit — deliberate, not debt)

**B — by design, never to be done:**

- No fix to FastInt64's `int64` overflow (by design; documented).
- No parallelism in the reference engine (single-threaded by contract).
- No FLINT / GMP / NTL / MPFR backend (pure-Python + optional Sage GSO only).
- No Sage / fplll wired as a reduction *engine* (comparison subjects only).

**D — long-term, documented in README:**

- FastInt64 remains an `int64` prototype: BKZ block size ≤ 80, very large
  dimensions / magnitudes are out of scope for the fast path (use the reference
  engine there).

**C — already handled elsewhere in this plan or externally:**

- C-8 (two multicore spec/plan docs untracked) is **done** — committed in
  `d0eca78`; not re-planned.
- C-9 (Singular not on PATH → one Sage test error) is **environment noise**,
  reproduced on the baseline; out of scope.
- C-7 (five multicore deferred minors) is folded into this plan as its final
  polish task.
