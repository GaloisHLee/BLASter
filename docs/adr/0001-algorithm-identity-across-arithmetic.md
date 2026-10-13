# ADR-0001: Preserve BLASter algorithm identity across arithmetic backends

- Status: Accepted
- Date: 2026-08-08

## Context

Current BLASter couples segmented scheduling and reduction decisions to int64
and binary64 kernels. Supporting checked integers and wider reals creates a risk
that a backend switch also changes predicates, rounding, traversal, or even the
reduction algorithm. That would make results impossible to attribute.

## Decision

The versioned algorithm specification defines BLASter's identity. FastInt64,
CheckedInt64, and Multiprecision implement the same segment partitions, offset
sequence, LLL/DeepLLL conditions, panel transformations, BKZ schedule,
enumeration order, and named tie rules.

Arithmetic escalation may revise an uncommitted uncertain decision after replay.
It may not choose a different condition or algorithm. A deliberate formula,
schedule, tie-rule, or stopping change requires a new algorithm-specification
version and separate benchmark evidence.

## Consequences

- Backends are comparable without pretending their byte-level trajectories
  always match the unchecked baseline.
- Every result records the algorithm version and a decision-transcript digest.
- Current inconsistencies, such as two tie rules and the BKZ volume argument,
  remain visible until an explicit algorithm-version review changes them.
- Backend-specific shortcuts are rejected when they alter committed decisions.

## Evidence

The algorithm boundary is defined in
[`docs/spec/blaster-algorithm.md`](../spec/blaster-algorithm.md); arithmetic and
trajectory equivalence are defined in
[`docs/spec/arithmetic-model.md`](../spec/arithmetic-model.md).
