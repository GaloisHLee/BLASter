# Architecture decision records

The M2 arithmetic design is governed by these accepted decisions:

1. [ADR-0001: Preserve BLASter algorithm identity across arithmetic backends](0001-algorithm-identity-across-arithmetic.md)
2. [ADR-0002: Keep external reducers outside the runtime backend model](0002-external-reducers-are-comparison-subjects.md)
3. [ADR-0003: Rebuild approximate state from exact checkpoints](0003-rebuild-real-state-from-exact-checkpoints.md)
4. [ADR-0004: Separate reduction from independent verification](0004-separate-reducer-and-verifier.md)
5. [ADR-0005: Stage multiprecision implementation by algorithm complexity](0005-multiprecision-delivery-order.md)
6. [ADR-0006: Synchronize upstream changes beneath the local development layer](0006-upstream-sync-boundary.md)
7. [ADR-0007: Deliver Stage 1 as an exact verifier, then a Sage reference reducer](0007-stage-1-exact-verifier-then-sage-reference.md)

"Accepted" means the decision governs work after M2. It does not imply that the
described multiprecision components already exist.
