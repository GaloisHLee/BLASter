# ADR-0007: Deliver Stage 1 as an exact verifier, then a Sage reference reducer

- Status: Accepted
- Date: 2026-10-13

## Context

The three reported defects — silent int64 overflow, a float64 rank
false-negative, and float64 size-reduction precision loss — all trace to the
unchecked int64 + binary64 fast path. M2 already scopes the arithmetic fix, but
a multiprecision reducer is gated behind Stage 1 (ADR-0005), which itself
requires a Sage reference implementation and a recorded review. The first
buildable artifact must therefore de-risk those findings without disturbing the
fast path, while remaining faithful to the original BLASter paper's algorithm.

## Decision

1. **Verifier first.** Land the independent verifier (ADR-0004) before any
   reducer change. For Stage 1 (LLL + classical panel) its reduction
   postconditions — weak-LLL and size reduction — are exactly checkable from an
   exact rational GSO, so the verifier needs no floating point or precision
   escalation. It uses arbitrary-precision integers for rank, `B0*U=B`, and
   `abs(det(U))=1`, and exact rational GSO for the postconditions.
2. **Dual exact-GSO backend.** The verifier's GSO uses a pure-Python `Fraction`
   implementation as the portable default (runs under bare `python`, no Sage
   dependency) and an optional Sage `gram_schmidt` accelerator invoked through
   `sage -python`. Structural checks are pure Python `int` only.
3. **Sage reference reducer, not Sage LLL.** Stage 1's reference reducer is an
   exact transcription of `SEGMENTED_LLL`, `LOCAL_LLL`, and `SIZE_PANEL` from
   `blaster-algorithm.md`, computing `round(mu)` exactly from integer Gram
   entries and reproducing both named tie rules. Sage's own `LLL`/`BKZ` are
   comparison oracles in tests only (ADR-0002), never the reference engine.
4. **Finalize the two Stage-1 ambiguities.** The two named tie rules and the
   local/global Lovász equality mismatch are accepted as documented
   compatibility behavior, with no algorithm-version change. BKZ's volume
   argument and DeepLLL's depth predicate are recorded but deferred to Stages 5
   and 4 respectively.
5. **Record the rank fix.** The float64 rank false-negative is already replaced
   in the worktree by an exact modular GF(p) rank check; this ADR records it as
   current behavior and reconciles M0/M1 accordingly.

## Consequences

- `make test` (bare `python`) exercises the pure-Python verifier and structural
  checks; a new `make test-sage` target runs Sage-dependent verifier and
  reference-reducer tests through `sage -python`.
- The verifier returns per-check verdicts; a deliberately corrupted reducer
  result must be rejected (corruption tests).
- The reference reducer emits a decision transcript; trajectory equivalence with
  FastInt64 is a compatibility test, not the correctness definition (ADR-0001).

## Evidence

- Stage 1 contract:
  [`docs/spec/stage-1-verifier-and-sage-reference.md`](../spec/stage-1-verifier-and-sage-reference.md)
- Algorithm: [`docs/spec/blaster-algorithm.md`](../spec/blaster-algorithm.md)
- Arithmetic: [`docs/spec/arithmetic-model.md`](../spec/arithmetic-model.md)
