# ADR-0003: Rebuild approximate state from exact checkpoints

- Status: Accepted
- Date: 2026-08-08

## Context

QR/GSO matrices accumulate rounding error. Widening an already rounded binary64
matrix does not recover the information lost from the exact integer basis.
Continuing a partially explored enumeration or a partially committed segmented
phase after increasing precision can mix incompatible decisions.

## Decision

`B0`, `B`, `U`, and scheduler position form exact checkpoints. `R`, panel solves,
projected norms, enumeration state, and diagnostic profiles are disposable
approximate epochs tied to one checkpoint and one real precision.

When a decision is uncertain or CheckedInt64 promotes its integer domain, the
session rolls back every provisional transform in that phase, restores the
checkpoint's scheduler state, converts exact `B` into the new real context, and
recomputes QR. Enumeration restarts from its root. Converting old `R` to a wider
type or resuming a mixed-precision search is forbidden.

## Consequences

- Precision recovery has a measurable replay cost and requires transactional
  phase commits.
- Concurrent segment results commit together or are all discarded.
- Event metadata can explain every escalation from a stable exact state.
- Implementations may optimize checkpoint storage with copy-on-write techniques,
  but observable rollback semantics remain unchanged.

## Evidence

Required checkpoint boundaries and escalation rules are specified in
[`docs/spec/arithmetic-model.md`](../spec/arithmetic-model.md).
