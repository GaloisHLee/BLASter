# ADR-0005: Stage multiprecision implementation by algorithm complexity

- Status: Accepted
- Date: 2026-08-08

## Context

Implementing every BLASter mode over a new arithmetic layer at once would make
arithmetic errors, scheduling errors, and algorithm misunderstandings difficult
to isolate. The modes form a natural dependency chain.

## Decision

After M0-M2 review, implementation proceeds through these gated stages:

1. ordinary non-segmented LLL with classical size reduction;
2. segmented LLL with alternating offsets and transactional phase commits;
3. recursive Seysen panel reduction;
4. DeepLLL with the current implemented depth predicate; and
5. segmented/progressive BKZ with pruned enumeration.

Each stage begins with a Sage reference implementation, specification examples,
exact structural verification, controlled-decision tests, M0 compatibility
measurements, and an explicit review before the next stage. A stage may share
interfaces established earlier but may not weaken their contracts.

## Entry gate for the Sage reference implementation

Stage 1 starts only after one recorded review confirms all of the following:

- algorithm reviewers accept version 0.1's matrix orientation, local/global
  equality mismatch, two named tie rules, DeepLLL depth predicate, and current
  BKZ volume formula as faithfully documented compatibility behavior;
- any decision to correct one of those behaviors is assigned a new
  specification version before reference code is written;
- arithmetic reviewers accept the exact-checkpoint boundary, 53-to-4096-bit
  precision profile, typed failures, and binary64-bit-pattern semantics for
  `delta`;
- the M0 corpus, commands, expected iteration/quality values, and external-tool
  role are frozen as comparison evidence;
- independent verifier inputs and ordinary-LLL/classical-panel postconditions
  are approved; and
- Stage 1 scope excludes segmentation, Seysen, DeepLLL, BKZ, parallelism,
  production optimization, and final native-backend selection.

Failure of a gate item keeps work in specification review. It does not authorize
an implementation-specific interpretation.

## Consequences

- The first implementation target deliberately excludes parallel scheduling,
  Seysen solves, deep insertions, and enumeration.
- BKZ cannot drive the arithmetic design prematurely.
- Performance optimization and final FLINT/GMP/NTL/MPFR selection follow
  correctness evidence rather than precede it.
- Existing FastInt64 code remains available as a baseline while reference work
  progresses.

## Evidence

The dependency boundaries are mapped in
[`docs/spec/implementation-map.md`](../spec/implementation-map.md). Review gates
and arithmetic contracts are defined in
[`docs/spec/arithmetic-model.md`](../spec/arithmetic-model.md).
