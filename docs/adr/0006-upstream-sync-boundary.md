# ADR-0006: Synchronize upstream changes beneath the local development layer

- Status: Accepted
- Date: 2026-08-13

## Context

The official BLASter repository advanced eleven commits beyond this checkout's
Git baseline while the local fork carried an uncommitted package/API/validation,
test, benchmark, and specification layer. A naive fast-forward would have
discarded local work; copying only selected upstream hunks would have left the
core and Python semantics ambiguous.

## Decision

The official upstream `main` commit
`68225ece2e10d6f50a5a811de63d97583a987dc6` is the new algorithm and package
base. Its core behavior is adopted, including:

- `blaster._core` as the Cython extension name;
- the shared `_propagate_update` segmented-commit routine;
- input-reading and stdio fixes;
- removal of the deprecated `Eigen::initParallel()` call;
- the weak-LLL predicate without a modulo step; and
- `beta < 40` behavior that runs DeepLLL-4 instead of a pruned BKZ tour.

Local development changes are reapplied on top of that base as a separate
layer: strict int64 input/parameter validation, `reduce_basis`, Sage matrix
preservation, the robust fplll parser, deterministic tests and examples,
benchmark notes, and the M0/M1/M2 specification documents. Packaging metadata
is consolidated in `pyproject.toml`, while `setup.py` remains responsible for
the native extension.

The worktree is synchronized without creating a branch commit. The `HEAD`
reference remains at the historical local baseline; upstream content and local
changes are visible as working-tree changes.

## Consequences

- Upstream behavior changes are not silently reverted by a later packaging
  cleanup.
- Local API documentation must state when a behavior differs from upstream.
- Package version moves to upstream `1.0`; historical 0.1.0 benchmarks are
  retained as pre-sync evidence.
- The M0 baseline and implementation map record both the historical baseline
  and the post-sync state.
- A future commit or merge can be reviewed against the upstream
  `upstream/main` reference without repeating the conflict analysis.

## Evidence

- Upstream provenance: [`docs/spec/current-behavior-baseline.md`](../spec/current-behavior-baseline.md)
- Algorithm semantics: [`docs/spec/blaster-algorithm.md`](../spec/blaster-algorithm.md)
- Code map: [`docs/spec/implementation-map.md`](../spec/implementation-map.md)
