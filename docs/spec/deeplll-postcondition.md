# DeepLLL postcondition and `verify_reduction(mode="deeplll")`

Status: accepted design (C-2). Version 0.1. Date 2026-10-14.

This document defines the exact, checkable postcondition of BLASter's
deep-LLL variant (the depth predicate implemented in `_deeplll_local`) and the
verifier contract that certifies it. It is the spec that C-2 is implemented
against: read it before changing `verify_reduction`.

## 1. What deep-LLL is

`depth > 0` selects the *deep-LLL* variant in place of ordinary local LLL.
After local LLL, `_deeplll_local` walks `k` and deep-inserts `b_k` at the
earliest `i < k` satisfying the depth predicate

```
(i < depth) or (i >= k - depth)
```

and

```
||pi_i(b_k)||^2 <= delta * ||b_i*||^2 ,
```

via adjacent swaps, where `pi_i(b_k)` is the projection of `b_k` onto the
orthogonal complement of `span(b_0, ..., b_{i-1})`. `depth` is a non-negative
integer; `depth = 0` dispatches to plain LLL (no deep condition). See
`src/blaster/reference.py:_deeplll_local` and `core/lattice_reduction.cpp:_deeplll_reduce`.

The name is *deep LLL* (Schnorr–Euchner deep insertion). It is **not** full
Schnorr–Euchner deep-LLL: it never evaluates the global potential
`sum_i ||b_i*||^2` to *choose* an insertion depth — it inserts at the earliest
`i` in the window whose test fires, exactly as the FastInt64 engine does.

## 2. The exact postcondition

Two invariants the deep-LLL variant guarantees — and therefore the only ones an
independent verifier can certify:

1. **Size reduction** (the same predicate as the classical and Seysen panels):

   > For every `0 <= i < j < n`, `abs(mu[i][j]) <= 1/2`,
   > equivalently `2 * abs(<b_j, b*_i>) <= r_i`.

2. **The deep condition**, for every `k` and every in-window `i < k`:

   > `||pi_i(b_k)||^2 > delta * ||b_i*||^2`,
   > where `i` is in the window iff `(i < depth) or (i >= k - depth)`.

   In exact rational GSO coefficients, `||b_i*||^2 = r_i` and

   > `||pi_i(b_k)||^2 = r_k + sum_{t=i}^{k-1} mu[t][k]^2 r_t`,

   i.e. `sum_{t=i}^{k} mu[t][k]^2 r_t` with the standard `mu[k][k] = 1`;
   `rational_gso` stores `mu[k][k] = 0`, so the diagonal term is `r_k` itself.

The inequality is **strict** (`>`): `_deeplll_local` inserts only on
`proj <= delta * r_i`, so a fixed point has `proj > delta * r_i` for every
in-window `i`. The adjacent pair `i = k-1` is always in the window when
`depth >= 1` (because `k-1 >= k - depth`), so the deep condition implies the
strict weak-LLL condition on every adjacent pair; weak-LLL is therefore
**computed and reported but not gated** into `mode="deeplll"` — it is a
consequence, not a separate requirement.

## 3. Verifier contract

`verify_reduction(B0, B, U, delta, mode="deeplll", depth=1, gso=...)` runs the
shared structural checks (which raise `VerificationFailed` on failure) and then
reports the reduction predicates as flags on a `VerificationReport`:

- Structural (raise on failure, all modes): full column rank of `B0` over `Q`,
  `B0 * U == B` exactly, `abs(det(U)) == 1` exactly.
- `size_reduced_ok`: `abs(mu[i][j]) <= 1/2` for all `i < j`, exact.
- `deep_ok`: the deep condition of §2 with the given `depth`, exact.
- `weak_lll_ok`: strict weak-LLL, exact — **computed and reported but not
  gated** for this mode.
- `verified = size_reduced_ok and deep_ok` for `mode="deeplll"`.

`depth` is a keyword used only by `mode="deeplll"`: it must be a non-negative
integer, and for this mode it must be at least `1` (a `depth=0` deep-LLL is
plain LLL with an empty window and is meaningless here). `mode` is validated:
`"lll"`, `"seysen"`, and `"deeplll"` are accepted; any other value raises
`ValueError`.

Contrast with `mode="lll"`, where `verified = weak_lll_ok and size_reduced_ok`.

## 4. Worked example (why the mode is not "seysen")

Columns `b0 = (100, 0)`, `b1 = (0, 1)`; `B0 = [[100, 0], [0, 1]]`, `U = I`.

- `r0 = 10000`, `r1 = 1`, `mu[0][1] = 0`.
- size-reduced: `|mu[0][1]| = 0 <= 1/2` — **holds**.
- deep condition (`depth = 1`): `k = 1`, the only `i = 0` is in the window
  (`0 < 1`). `||pi_0(b_1)||^2 = r1 + mu[0][1]^2 r0 = 1`; `delta * r0 = 9900`;
  `1 > 9900` is false — **fails**.

So `mode="seysen"` certifies this basis (`verified=True`, it is size-reduced)
while `mode="deeplll"` does not (`deep_ok=False`, `verified=False`). This is
the same two-column pair the Seysen spec uses, and it doubles as the corruption
fixture below.

## 5. Corruption property (drives the test)

The two-column basis above is already a clean rejection fixture: it passes
every structural check (`B0 * U == B`, `det(U) = 1`) and is size-reduced, yet
fails the deep condition. The acceptance test asserts `deep_ok == False` and
`verified == False` **without raising** under `mode="deeplll"`, and asserts a
genuine deep-LLL output (`lll_reduce(..., depth=...)`) certifies
`verified == True`. Adding `z * col_i` to `col_j` (`i < j`, in-window) on both
`B` and `U` is a unimodular column operation that keeps the structural checks
passing while shifting `mu[i][j] += z`; the two-column fixture is the
`z`-independent, deterministic version of that corruption.

## 6. Out of scope

- No global-potential evaluation, no Schnorr–Euchner *selection* of the
  insertion depth, no pruning/enumeration.
- No certification of *which* insertion depth or trajectory was chosen.
- No change to the FastInt64 engine or to `mode="lll"` / `mode="seysen"`
  semantics.
