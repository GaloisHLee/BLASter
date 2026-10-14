# Stage 5: BKZ pruned enumeration — exact transcription spec

- Status: accepted design (D-2). Version 0.1. Date 2026-10-14.

> 中文摘要：本文档定义精确参考引擎的 BKZ 阶段（Stage 5）。它把 FastInt64 引擎的
> `bkz_reduce`/`svp`/`enumeration`（`core/lattice_reduction.cpp`、`core/enumeration.cpp`、
> `core/enumeration.hpp`、`core/pruning_params.cpp`）逐行转写成任意精度 `Fraction`，
> 唯一允许的浮点来源是 GH 半径的超越函数（`lgamma`/`exp`/`log`/`pi`），并且这些值在进入
> 枚举之前被钳成精确 `Fraction`。枚举内部的每一处比较与范数计算都是精确有理数算术。

## 1. What BKZ (Stage 5) is

The FastInt64 engine reduces a basis by a **progressive-block tour** of *pruned
Schnorr–Euchner enumeration*:

1. **Pre-reduce** the whole basis with `deeplll_reduce(delta, depth=4)`.
2. **`beta <= 2`**: stop — the result is exactly DeepLLL-4 (no `svp` is called).
3. **`2 < beta <= n`** (BKZ tour): for each block start `i` in `[0, n-beta]`, run
   `svp(i, beta)`.
4. **`beta > n`** (HKZ tour): for each `i` in `[0, n-2]`, run `svp(i, n-i)`.

Each `svp(i, w)` enumerates the `w`-dimensional sublattice spanned by
`b_i, …, b_{i+w-1}` for a short nonzero vector under the GH radius with GNR
pruning coefficients, then — if one is found — inserts it at position `i` by a
single column combination followed by `j` adjacent swaps, and finally
re-deep-reduces the window `[0, min(i+w, n))` with `deeplll_reduce(delta, 4)`.

The exact reference engine reproduces this decision-by-decision, with every
GSO coefficient, norm, and comparison in exact `Fraction` arithmetic. It never
calls Sage `BKZ()`/`LLL()` as the engine (constraint 1), and it never uses
`sqrt`, `llround`, or `numpy.linalg` on the exact path (constraint 2).

## 2. Exact input mapping (the "no floating point" crux)

The C++ enumeration operates on the QR factor `R` with `FT = double` and
`ZZ = long long` (`core/types.hpp`). The exact transcription maps every
`FT`/`ZZ` quantity to `Fraction`/`int` **without** any floating-point step:

| C++ quantity | exact replacement | why exact |
|---|---|---|
| `muT[i][j] = R[i][j] / R[i][i]` (i<j) | `mu[i][j]` from `rational_gso` | GSO coefficient is `⟨b_j,b*_i⟩/‖b*_i‖²`, an exact rational |
| `risq[i] = R[i][i] * R[i][i]` | `r[i]` from `rational_gso` | `‖b*_i‖²`, an exact rational |
| `pr[i]` (a 4-decimal literal) | `Fraction(v, 10000)` | the literal is an exact decimal |
| `round(ci)` then `(ZZ)yi` | `nearest_core(ci)` | `round` = tie away from zero = `nearest_core` (constraint 5) |
| `yi = ci - yi` | `Fraction ci - int xi` | exact subtraction |
| `_A`, `_AA[k]`, `_l[k]`, `_sigT[i][j]`, `_c[i]` | `Fraction` | exact |
| `_x`, `_Dx`, `_D2x`, `_r`, `_sol` | `int` | exact |

The single rounding step `round(ci)` followed by `(ZZ)yi` collapses to
`xi = nearest_core(ci)` (an `int`), and `yi = ci - xi` (a `Fraction`). `ci`
is a `Fraction`, so `nearest_core` (which is `sign(a)·⌊(2·|a|+b)/(2b)⌋` for
`x = a/b`, ties away from zero) is the exact, certified center.

The C++ pads the enumeration to `MAX_ENUM_N = 80` dimensions with inert
virtual coordinates (`risq[i] = 2·risq[0]`, `pr[i] = 1.0`, `muT ≡ 0`, so every
virtual `x[i]` stays 0). That padding is a fixed-size-stack implementation
detail: the exact transcription recurses over exactly `n = w` real dimensions
and is semantically identical.

## 3. The GH radius and pruning coefficients (controlled approximate)

Two scalars enter the enumeration from outside exact rational arithmetic:

- **`log_volume`** — `sum_{j<w} log(RSQ(i+j, i+j))` = `sum_{j<w} log(r[i+j])`,
  computed as `math.log(float(r[i+j]))` (binary64).
- **`_gh_squared(dim, log_volume)`** — `math.exp(2.0*(math.lgamma(dim/2.0+1.0) + log_volume)/dim) / math.pi`.
- **`_safe_gh_squared(dim, log_volume)`** — `gh2 = _gh_squared(...)`;
  `factor = max(1.05, min(2.0, 1.0 + 4.0/dim))`;
  returns `Fraction(*(gh2 * factor * factor).as_integer_ratio())`.

These are the **only** floating-point sources, and per `docs/spec/arithmetic-model.md`
they are a *controlled approximate operation* (error-bounded): they are computed in
binary64 and then snapped to an exact `Fraction` via the same
`Fraction(*value.as_integer_ratio())` escape hatch used for a `float` `delta`.
A controlled run may differ from the FastInt64 engine only where the latter made
an uncertified binary64 decision at the radius boundary; the enumeration **decisions**
are then exact and deterministic for a given snapped radius.

The enumeration radius is `expected = min((1023/1024)·r[i], _safe_gh_squared(w, log_volume))`.
The C++ `1023.0/1024` is exactly representable; the exact transcription uses the
exact `Fraction(1023, 1024)` (the residual difference is the documented controlled
approximation of `r[i]` and the product, already covered by the model).

The pruning coefficients come from `core/pruning_params.cpp`, the table used by
`get_pruning_coefficients(block_size) = &pruning_params[(block_size+1)/2][0]`.
Index `k` holds the BKZ-`2k` row; a block of size `w` consumes `pr[0..w-1]` =
`PRUNING_PARAMS[(w+1)//2][:w]` as `Fraction(v, 10000)`. Rows are converted to
integer ×10000 form **programmatically** from the source (never hand-typed).

## 4. The enumeration recursion (exact transcription)

State (all `Fraction` except the integer arrays): `A` (best radius), `AA[k] =
A·pr[k]`, `x`, `Dx`, `D2x` (init `-1`), `c`, `r` (init `n-1`, the "last-changed"
index), `l[0..n]` (init 0), `sigT` (init 0), `sol` (init 0).

`enumeration(mu, r, pr, radius)` initializes `A = radius`, runs `_recur(n-1)`,
and returns `(sol, A if A < radius else Fraction(0))` — mirroring the C++
`return (_A < enumeration_radius) ? _A : 0.0` and the `_sol[i] != 0` virtual check
being vacuous with no padding.

**`_recur_igen(i)`, `i >= 1`** (transcribes `enumerate_recur<i_tag<i>>`):

```text
if r[i] > r[i-1]: r[i-1] = r[i]
ci = sigT[i][i]
xi = nearest_core(ci);  yi = ci - xi
li = l[i+1] + yi*yi*r[i]
if li > AA[i]: return
Dx[i] = D2x[i] = (1 if yi >= 0 else -1)
c[i] = ci;  x[i] = xi;  l[i] = li
for j in r[i-1] down to i:  sigT[i-1][j-1] = sigT[i-1][j] - x[j]*mu[i-1][j]
loop:
    _recur(i-1)
    if l[i+1] == 0: x[i] += 1
    else: x[i] += Dx[i];  D2x[i] = -D2x[i];  Dx[i] = D2x[i] - Dx[i]
    r[i-1] = i
    yi2 = c[i] - x[i];  li2 = l[i+1] + yi2*yi2*r[i]
    if li2 > AA[i]: return
    l[i] = li2
    sigT[i-1][i-1] = sigT[i-1][i] - x[i]*mu[i-1][i]
```

**`_recur_i0` (i = 0)** is identical but omits (a) the top `r[0] > r[-1]` update,
(b) the projection loop, and (c) the in-loop `r[i-1] = i` and `sigT[i-1][i-1]`
update — the C++ `enumerate_recur<i_tag<0>>` specialization.

**`_recur(-1)` (base, `i_tag<-1>`)**:

```text
if l[0] > A or l[0] == 0: return
for j: sol[j] = x[j]
A = l[0];  _update_AA()          # AA[k] = A*pr[k] for every k
```

`l[i]` is the exact partial norm of the projection onto `span(b*_i,…,b*_{n-1})`;
`l[i+1] == 0` is the exact "nothing committed above" first-visit test that turns
the Schnorr–Euchner zig-zag into a plain `x[i] += 1` on the first descent.

## 5. The `svp` post-step and the BKZ/HKZ tour

`svp(cols, Ucols, i, w, delta, tr, gso)`:

```text
mu, r = gso(cols)
mu_b[a][b] = mu[i+a][i+b];  r_b[a] = r[i+a]
log_volume = sum(math.log(float(r[i+a])) for a in range(w))
expected = min(Fraction(1023,1024)*r[i], _safe_gh_squared(w, log_volume))
pr = get_pruning_coefficients(w)
sol, norm = enumeration(mu_b, r_b, pr, expected)
if norm > 0:
    j = w-1;  while j > 0 and sol[j] == 0: j -= 1
    if j > 0 and sol[j] == -1: sol[0..j] = -sol[0..j]
    if j > 0 and sol[j] == 1:
        for k in range(j):
            if sol[k]:  add sol[k]·(col i+k) to col i+j (cols and Ucols, record "add")
        for p in range(j-1, -1, -1):  swap cols i+p and i+p+1 (record "swap")
_deeplll_local(cols, Ucols, 0, min(i+w, n), delta, 4, tr, gso)   # unconditional
```

The `sol[j]` sign canonicalization, the `alter_basis`/`swap_basis_vectors` insertion
(written as the existing `_add_col`/`_swap_cols` helpers), and the unconditional
deep-LLL-4 post-step are the exact C++ `svp` transcription.

`bkz_reduce(B0, delta=0.99, *, beta, gso=None)`:

```text
parse B0; full-rank check; d = delta_to_fraction(delta); beta = _beta_param(beta)
cols, Ucols = identity; tr = []
_deeplll_local(cols, Ucols, 0, n, d, 4, tr, gso)   # whole basis, depth 4
if beta <= 2: return (cols, Ucols^T, tr)
if beta > n:  for i in range(n-1):   svp(i, n-i)   # HKZ
else:         for i in range(n-beta+1): svp(i, beta)  # BKZ
return (cols, Ucols^T, tr)
```

`beta` is validated as an integer with `2 <= beta <= 80` (the enumeration/pruning
cap; `beta > n` selects the HKZ branch, so `beta <= n` is *not* required).

## 6. The exact postcondition (verifier contract)

After any tour, the **last** operation on the whole basis is a
`deeplll_reduce(delta, 4)`:

- `beta <= 2`: the initial whole-basis deep-LLL-4 is the only reduction.
- BKZ tour (`2 < beta <= n`): the last `svp` is at `i = n-beta`, `w = beta`,
  `min(i+w, n) = n`.
- HKZ (`beta > n`): the last `svp` is at `i = n-2`, `w = 2`, `min(n-2+2, n) = n`.

Therefore the certifiable BKZ postcondition is **size-reduced AND
deep-LLL(delta, depth=4) on the whole basis** — no weak-LLL gating, because the
deep condition implies the relevant inequalities. `verify_reduction(mode="bkz")`
gates `verified = size_reduced_ok and deep_ok` with `deep_lll_holds(mu, r, d, 4)`
(depth fixed at 4, independent of the user `depth` kwarg).

## 7. Worked example (postcondition)

Take `B0 = [[1,0,0],[0,1,0],[0,0,1]]` (3×3 identity, columns `e_0, e_1, e_2`).
`bkz_reduce(B0, beta=2)` = deep-LLL-4 whole basis = identity (already reduced).
GSO: `r = [1, 1, 1]`, `mu` all zero. `verify_reduction(B0, B, U, mode="bkz")`:

- `size_reduced_ok`: `abs(mu[i][j]) <= 1/2` for every `i < j` → `True`.
- `deep_ok = deep_lll_holds(mu, r, d, 4)`:
  - `k=1`: `proj = r[1] = 1`; window `{i < 4}` → `i=0`; `1 > d·1` (`d=0.99`) → pass.
  - `k=2`: `proj = r[2] = 1`; `i=0,1` both in window; `1 > d·1` each → pass.
- `verified = True`, `mode = "bkz"`, `delta = Fraction(99, 100)`.

A deliberately corrupted result (e.g. a non-size-reduced `mu[0][1] = 3/4`) fails
`size_reduced_ok`; a basis whose `proj <= d·r_i` for an in-window `i` fails `deep_ok`
— both report `verified = False` without raising.

## 8. Corruption property (drives the test)

The property the verifier must catch is a *local* postcondition violation that
preserves all *structural* facts: a valid unimodular transform `U` (`B0·U == B`,
`det(U) = ±1`) whose `B` is *not* deep-LLL-4. The verifier must reject it on
`deep_ok` (or `size_reduced_ok`) while `structural_ok`, `product_ok`,
`unimodular_ok` all remain `True`. This is the test that pins the depth-4 gating.

## 9. Out of scope

- Do **not** change the FastInt64 engine's arithmetic, nor route any `engine=` to
  Sage/fplll (constraint 9, ADR-0002/0005).
- Do **not** introduce parallelism into the reference engine; the transcription is
  a deterministic sequential tour (parallelism is a FastInt64 performance property).
- Do **not** add dependencies beyond Python stdlib + NumPy + the isolated Sage
  accelerator. The pruning table is embedded, not read at runtime.
- Do **not** alter the frozen M0 baseline expectations without a spec version bump.
- The verifier/reference stay `float`-free except for the GH radius and the
  `delta` dyadic path already documented in `arithmetic-model.md`.
