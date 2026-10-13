# Stage 1: independent verifier and Sage reference reducer

Version: 0.1 (Stage 1 delivery). Status: accepted design, not yet implemented.

This document is the delivery contract for ADR-0005 Stage 1 and ADR-0004. It
defines the two components built before any reducer change: an independent exact
verifier and an exact Sage reference reducer for ordinary LLL with classical
panel reduction. It does not change M0/M1/M2; it refines M2's generic
"controlled precision" verifier step for the Stage 1 postconditions, which are
exactly checkable.

## Purpose and scope

- **Verifier**: consumes a reducer result `(B0, B, U, mode, parameters)` as
  untrusted data and certifies (or rejects) the structural and Stage 1 reduction
  contracts using arbitrary-precision integers and exact rational GSO. It is
  segmentation-agnostic: it checks the output contract regardless of how the
  basis was produced, so it can audit the current FastInt64 output directly.
- **Sage reference reducer**: an exact, slow transcription of the BLASter
  algorithm for the Stage 1 mode — `SEGMENTED_LLL` with a single segment
  (`segment_size = rank`) and `SIZE_PANEL`. It produces a certified baseline and
  a decision transcript for trajectory comparison. It never calls Sage's
  `LLL`/`BKZ` as its engine.
- **Out of scope** (per ADR-0005): segmentation and concurrency, Seysen, DeepLLL,
  BKZ, parallelism, production optimization, and final native-backend selection.

## Spec finalization

Two flagged ambiguities affect Stage 1 and are accepted as documented
compatibility behavior with no algorithm-version change:

- **Two named tie rules**: `nearest_core` (tie away from zero; local LLL) and
  `nearest_panel` (tie to even; classical panel). Both are reproduced exactly.
- **Local/global Lovász equality mismatch**: local LLL accepts the non-strict
  condition; the global weak-LLL checker uses the strict condition. Both are
  preserved.

Two further ambiguities are recorded but deferred: the BKZ volume argument
(Stage 5) and the DeepLLL depth predicate (Stage 4).

## The independent verifier

### Inputs (untrusted)

`B0`, `B`, `U`, the mode (`LLL + classical panel` for Stage 1), `delta`, and the
effective parameters. The verifier reads these as plain data; it has no access
to reducer scratch buffers and places no trust in reducer claims (ADR-0004).

### Structural checks — exact, pure Python `int`

1. Parse all entries into arbitrary-precision integers (no `long long`).
2. Exact rank over `Q` of `B0` via fraction-free Gaussian elimination; require
   full column rank `n`.
3. `B0 * U == B` exactly (arbitrary-precision integer product and comparison).
4. `det(U)` exactly via Bareiss; require `abs(det(U)) == 1`.

These four checks are dependency-free and run under bare `python`.

### Reduction postconditions — exact rational GSO

The R-factor entries are GSO quantities. Write the exact Gram–Schmidt basis as
`b*_i = b_i - sum_{j<i} mu[j,i] * b*_j` with exact rational
`mu[j,i] = <b_i, b*_j> / <b*_j, b*_j>` and `r_i = <b*_i, b*_i>`. Then
`R[i,i]^2 = r_i` and `R[i,i+1] = mu[i,i+1] * R[i,i]` hold, so the two Stage 1
postconditions reduce to exact rational comparisons:

- **weak-LLL(delta)** (strict): for every `i`,
  `r_{i+1} + mu[i,i+1]^2 * r_i > delta * r_i`.
- **size-reduced** (non-strict): for every `i < j`, `abs(mu[i,j]) <= 1/2`,
  equivalently `abs(2 * <b_j, b*_i>) <= r_i`.

Every quantity is rational and every comparison is exact. No floating point and
no precision escalation are involved in the Stage 1 verifier.

### delta semantics

`delta` is normalized exactly as the reducer normalizes it: to the exact dyadic
rational of its IEEE-754 binary64 value (`float(delta).as_integer_ratio()`). The
verifier does not reinterpret the decimal literal (M2 "binary64-bit-pattern
semantics for `delta`"). Consequently, on inputs where FastInt64's binary64
decisions erred, the verifier may legitimately report weak-LLL as failed; that
signal is the point of finding #3, not a verifier bug.

### Dual exact-GSO backend

- **Portable default**: pure-Python `fractions.Fraction` GSO. Denominator growth
  is acceptable at verifier scale; no Sage dependency.
- **Accelerator**: Sage `matrix(ZZ).gram_schmidt(orthonormal=False)` when running
  under `sage -python`, as an independent second implementation.

The two backends must agree on every verdict; a disagreement is a verifier bug,
never a reducer verdict.

### Verdicts and failure

The verifier returns per-check verdicts (structural, weak-LLL, size-reduced)
rather than one boolean. Any failed structural check yields `VerificationFailed`
with evidence and leaves the result unverified (ADR-0004). Corruption tests must
demonstrate rejection of (a) `B0*U != B`, (b) `abs(det(U)) != 1`, and (c) a
non-weak-LLL or non-size-reduced output.

## Stage 1 Sage reference reducer

### Scope

An exact transcription of `SEGMENTED_LLL(B0, B, U, segment_size=rank, delta,
depth=0, panel_mode=CLASSICAL)` from `blaster-algorithm.md`, including the
half-offset toggle (`offset = floor(rank/2)` first, then `0`, alternating) and
the two-QR-per-iteration structure. It emits `(B, U)` plus a decision transcript.

### Exact decision arithmetic

The reducer maintains exact rational GSO `(mu, r)` as the arithmetic
representation of `R`. The Givens rotation used by the floating fast path is not
needed: under exact arithmetic, recomputing GSO after an operation yields the
same `(mu, r)` that exact Givens would, and every decision is a function of
`(mu, r)` alone.

- Size coefficient: `z = nearest_core(-mu[i,k])` for local LLL, `nearest_panel`
  for the classical panel.
- Local Lovász (non-strict): `delta*r_{k-1} <= r_k + mu[k-1,k]^2*r_{k-1}`.
- Weak-LLL (strict) for the global stop, as above.

Exact nearest-integer operators on a rational `a/b` (`b > 0`):

- `nearest_core(a/b) = sign(a) * floor((2*abs(a) + b) / (2*b))` — tie away from
  zero;
- `nearest_panel(a/b)`: exact quotient and remainder; on an exact half-integer,
  round the magnitude to the nearest even integer (round-half-to-even).

All comparisons and roundings are exact; there is no float, no `sqrt`, and no
`llround`.

### Decision transcript

Each committed operation is recorded in order as `(op, i, j, z, tie_rule)` for
column additions and `(swap, i, i+1)` for swaps, plus a per-iteration
`(offset, stop_decision)`. The transcript enables trajectory comparison with
FastInt64 on inputs where binary64 makes no decision error.

### Non-goals

No concurrency, no Seysen/DeepLLL/BKZ, no Sage `LLL`/`BKZ` as the engine, no
optimization, no backend selection. Sage's reducers appear only as comparison
oracles in tests.

## Engineering compatibility

- `make test` (bare `python`) runs the pure-Python verifier, the structural
  checks, and the existing NumPy tests.
- `make test-sage` runs Sage-dependent tests — the `gram_schmidt` accelerator,
  the reference reducer, and Sage cross-check oracles — through `sage -python`.
- Sage-importing tests are isolated in a Sage-only module so bare-python test
  discovery skips them; this fixes the current
  `test_reduce_basis_preserves_sage_matrix_type` failure under bare `python`.

## Acceptance criteria

1. The verifier certifies the current FastInt64 output on the M0 corpus
   (structural + weak-LLL + size-reduced all pass on exact-friendly inputs).
2. The verifier rejects each corruption class in "Verdicts and failure".
3. The verifier certifies a wide-dynamic-range HNP basis (the findings #2/#3
   case) independently of the float fast path.
4. The Sage reference reducer's output is accepted by the verifier on
   deterministic examples.
5. On exact-friendly inputs, the reference reducer run with `segment_size=rank`
   matches FastInt64 run with `lll_size=rank` in decision transcript.
6. Half-integer unit tests cover both tie rules.
