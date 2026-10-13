# ADR-0004: Separate reduction from independent verification

- Status: Accepted
- Date: 2026-08-08

## Context

The current reducer and tests often verify relations using the same int64 values
and numerical routines that produced the result. Shared overflow or floating
error can make a self-check pass without establishing the mathematical claim.
Different reduction modes also promise different postconditions.

## Decision

The reducer emits a result, completion status, algorithm/arithmetic metadata,
and decision transcript. A separate verifier treats those objects as untrusted.
It uses arbitrary-precision exact arithmetic for rank, `B0*U=B`, and
`abs(det(U))=1`, then independently recomputes controlled-real state and checks
only the selected mode's specified reduction postconditions.

The verifier has its own precision ceiling and error reporting. Its code does
not call reducer predicates as the sole implementation of a check. Another
reducer's output is not a verification oracle.

## Consequences

- `verified` becomes an evidence-bearing status, not a reducer assertion.
- FastInt64 results remain explicitly unverified until the independent component
  accepts them, and undefined signed overflow still cannot be retroactively
  ruled out solely by matching output matrices.
- Verification can be developed and tested before a multiprecision reducer.
- Corruption tests must demonstrate rejection by the verifier.

## Evidence

The verifier protocol is specified in
[`docs/spec/arithmetic-model.md`](../spec/arithmetic-model.md), while mode-specific
postconditions are in
[`docs/spec/blaster-algorithm.md`](../spec/blaster-algorithm.md).
