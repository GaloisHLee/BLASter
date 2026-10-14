# Seysen-panel postcondition and `verify_reduction(mode="seysen")`

Status: accepted design (B-2). Version 0.1. Date 2026-10-14.

This document defines the exact, checkable postcondition of BLASter's recursive
Seysen panel and the verifier contract that certifies it. It is the spec that
B-2 is implemented against: read it before changing `verify_reduction`.

## 1. What the Seysen panel is

`use_seysen=True` selects the *recursive Seysen panel* in place of the classical
size-reduction panel. On the QR factor `R = D * M` (`D = diag(sqrt(r_i))`,
`M` unit upper-triangular with `M[a][b] = mu[a][b]`), the panel walks
`__reduction_ranges(n)` bottom-up: two-column leaves round `mu[i][i+1]`, and
each cross panel `[j:k)` against `[i:j)` computes

```
U12' <- nearest_panel(-inverse(S11) * S12') entrywise
S12  <- S12' + S11 * U12'
```

The name is a *recursive Seysen-style* transformation. It is **not** full
Seysen reduction (Seysen 1993): the panel never computes the dual basis, never
evaluates the Seysen measure `S(B) = sum_i ||b_i||^2 ||b_i^*||^2`, and never
performs Seysen swaps. See `docs/spec/blaster-algorithm.md` ("the recursive
Seysen name is stronger than its checked postcondition").

## 2. The exact postcondition

The only invariant the Seysen panel guarantees — and therefore the only one an
independent verifier can certify — is **size reduction**:

> For every `0 <= i < j < n`, `abs(mu[i][j]) <= 1/2`,
> equivalently `2 * abs(<b_j, b*_i>) <= r_i`.

This is the same predicate as the classical panel's `is_size_reduced`, evaluated
over the exact rational GSO coefficients `mu` (there is no floating point; see
`_exact.rational_gso` and `_exact.is_size_reduced`). The Seysen and classical
panels choose *different* size-reduced representatives; the verifier certifies
the invariant both share, never *which* representative was chosen.

The enclosing `SEGMENTED_LLL` loop terminates on weak-LLL. That condition is
imposed by the loop, not by the panel, so it is **not** part of the panel's
postcondition and must not be gated into `mode="seysen"`.

## 3. Verifier contract

`verify_reduction(B0, B, U, delta, mode="seysen", gso=...)` runs the shared
structural checks (which raise `VerificationFailed` on failure) and then reports
the reduction predicate as flags on a `VerificationReport`:

- Structural (raise on failure, all modes): full column rank of `B0` over `Q`,
  `B0 * U == B` exactly, `abs(det(U)) == 1` exactly.
- `size_reduced_ok`: `abs(mu[i][j]) <= 1/2` for all `i < j`, exact.
- `weak_lll_ok`: strict weak-LLL, exact — **computed and reported but not
  gated** for this mode.
- `verified = size_reduced_ok` for `mode="seysen"`.

Contrast with `mode="lll"`, where `verified = weak_lll_ok and size_reduced_ok`.
`mode` is validated: only `"lll"` and `"seysen"` are accepted; any other value
raises `ValueError`.

## 4. Worked example (why the mode is not "lll")

Columns `b0 = (100, 0)`, `b1 = (0, 1)`; `B0 = [[100, 0], [0, 1]]`, `U = I`.

- `r0 = 10000`, `r1 = 1`, `mu[0][1] = 0`.
- size-reduced: `|mu[0][1]| = 0 <= 1/2` — **holds**.
- weak-LLL: `r1 + mu[0][1]^2 r0 = 1`, `delta * r0 = 9900`; `1 > 9900` is false —
  **fails**.

So `mode="seysen"` certifies this basis (`verified=True`) while `mode="lll"`
does not. The panel's leaf `U[0][1] = nearest_panel(-mu[0][1]) = 0` leaves the
pair untouched, so this is a valid Seysen-panel fixed point.

## 5. Corruption property (drives the test)

Adding `z * col_i` to `col_j` (`i < j`) on both `B` and `U`'s corresponding
columns keeps `B0 * U == B` and `det(U) = +/-1` (it is a unimodular column
operation), so the structural checks still pass. On the GSO coefficients it
shifts `mu[i][j] += z` exactly. With `z = 2` the new
`abs(mu[i][j]) >= 3/2 > 1/2`, so size reduction is deterministically broken
while structure remains valid. This is the corruption the acceptance test uses:
the verifier must report `size_reduced_ok == False` and `verified == False`
without raising.

## 6. Out of scope

- No dual-basis computation, no Seysen measure, no Seysen swaps.
- No certification of *which* size-reduced representative was produced.
- No change to the FastInt64 engine or to `mode="lll"` semantics.
