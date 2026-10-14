# BLASter 完善 (Refinement) Loop — Scope, Backlog, and Anti-Drift Plan

- Status: Active (governs the recurring `/loop` "将本Blaster 完善" work)
- Date: 2026-10-14
- Owner of record: the recurring loop cron (`*/10 * * * *`, prompt `完成 M-2 M-3 M-4 ，并且将本Blaster 完善`)

> 中文摘要：本文档是「完善 Blaster」循环的唯一计划来源，用于防治漂移。循环不再临时发挥，而是每次 tick 按本文档的待办清单顺序推进，逐项独立提交。硬性边界（见 §3）与「禁止清单」（见 §9）不可违背。

## 1. Why this document exists

The loop prompt is open-ended ("…并且将本Blaster 完善"). Open-ended work drifts.
This document is the loop's single source of truth for **what to do next**, **what
counts as done**, and **what must never be touched**. Every future tick picks the
first open item in §5, implements it under §6, and does not improvise outside §3/§9.

M-2, M-3, and M-4 are **complete**. The operative part of the prompt is therefore
"将本Blaster 完善", which this document scopes.

## 2. Goal and scope

**Goal (governing principle, verbatim):** 使用现在的 LLM 能力、Sage 能力、Blaster 原始论文
结合起来，做一个高级版本的 Blaster，**减少精度限制**，**增加工程兼容性**.

Two tracks, both in scope:

- **Precision (减少精度限制):** extend the exact/reference engine so BLASter's
  segmented algorithm is reproduced under arbitrary-precision arithmetic, not only
  the single-segment classical-LLL slice delivered in Stage 1.
- **Engineering compatibility (增加工程兼容性):** the exact/reference engine accepts
  the same input kinds (nested lists, NumPy, Sage), reports the same diagnostics
  (exact `get_profile`), and exposes the same `delta`/`gso`/orientation contracts
  as the FastInt64 engine — without ever calling Sage `LLL()`/`BKZ()` as the engine.

The FastInt64 engine remains the performance baseline; the exact/reference engine is
the correctness oracle and compatibility target. External reducers (Flatter, fplll)
remain comparison subjects only (ADR-0002).

## 3. Non-negotiable constraints (hard boundaries)

Violating any of these is drift, regardless of how "useful" the change feels:

1. **Reference reducer is a transcription, not a Sage call.** It must implement
   BLASter's `SEGMENTED_LLL` (and, per stage, Seysen/DeepLLL/BKZ) itself. Calling
   `sage.matrix.LLL()`/`BKZ()` (or fplll/fpylll) as the reduction engine is forbidden.
   Sage may only accelerate a primitive (e.g. `gram_schmidt`), never decide the algorithm.
2. **No floating point in the verifier or reference.** Weak-LLL/size-reduction checks
   and the reducer's decisions are exact `int`/`Fraction`; no `sqrt`, no `llround`,
   no `numpy.linalg` on the exact path. A `float` only appears where the FastInt64
   engine's binary64 semantics are being *mirrored* (e.g. the dyadic `delta` path).
3. **Arbitrary-precision everywhere in the exact path.** Every matrix entry parses to
   Python `int`; `long long`/NumPy int64 never participates in verifier/reference math.
4. **FastInt64 int64 overflow is by-design, not a gap to fix.** The FastInt64 engine
   is a signed-int64 baseline; do not "repair" it by widening silently. Correct
   large-entry behavior comes from the reference engine, not from mutating the fast one.
5. **Two named tie rules, reproduced exactly.** `nearest_core` (tie away from zero,
   local LLL) and `nearest_panel` (round-half-to-even, classical/Seysen panel). No site
   substitutes Python's built-in `round`.
6. **Local/global Lovász mismatch preserved.** Local LLL uses the non-strict test;
   the global stop uses the strict weak-LLL test. Do not "unify" them.
7. **`delta` contract.** A `float` keeps its exact binary64 dyadic
   (`Fraction(*delta.as_integer_ratio())`); an exact `Fraction`/Sage `Rational`/integer
   is preserved exactly (not rounded through `float`). Enforce `1/4 < delta < 1` exactly.
8. **Column orientation `B0 · U == B`; row API `T · A0 == A` with `T = U^T`.** Never
   mix orientations silently.
9. **Sage imports isolated.** `verify_sage.py` is the only module importing `sage.all`;
   `blaster/__init__.py` never imports it. Sage tests live in `tests/sage/` (no
   `__init__.py`) and run only via `sage -python`.
10. **`unittest`, not pytest.** `make test` (bare Python) and `make test-sage`
    (`sage -python`) are the only authoritative suites.
11. **Never `git add t.py`** (deliberately untracked scratch at repo root).
12. **TDD.** Production code requires a failing test first (RED → GREEN), both suites
    green, then an **independent commit** for the item.
13. **End git commit messages with** `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## 4. Authoritative marching order

ADR-0005 stages the multiprecision implementation by algorithm complexity. This is
the loop's marching order — Stage 1 is done; the loop advances 2 → 5, never out of order:

1. ~~ordinary non-segmented LLL, classical size reduction~~ — **Stage 1, DONE** (M-1…M-4)
2. **segmented LLL, alternating offsets, transactional phase commits** — next
3. **recursive Seysen panel reduction**
4. **DeepLLL, current depth predicate**
5. **segmented/progressive BKZ, pruned enumeration**

Each stage starts with a reference implementation + spec examples + exact structural
verification + controlled-decision tests + M0 compatibility measurement, then an
explicit review before the next stage. A stage may share earlier interfaces but never
weaken them.

## 5. Backlog

Each item is sized to **one loop tick** (~10 minutes): a failing test, minimal code,
both suites green, one independent commit. Items are ordered; a tick takes the first
open one. A `[done]` item is already committed.

### Track A — Stage 2: segmented LLL reference

- **A-1. `lll_reduce` gains a segment width (`lll_size`).** [done] When `lll_size < rank`,
  partition the columns into disjoint segments and run local-LLL per segment in a
  deterministic sequential order, committing each segment's transform before the next.
  *Acceptance:* on q-ary/random cases with `lll_size < n`, reference output equals
  FastInt64 `reduce(A, lll_size=..., delta, cores=1)` exactly, certified by
  `verify_reduction`. *Note:* the reference is a deterministic sequential transcription
  of a parallel algorithm — parallelism is a performance property, not a correctness
  one, and is out of scope for the exact engine.

- **A-2. Alternating-offset toggle, pinned by transcript.** [done] Reproduce the first
  half-offset-then-zero sequence from `blaster.py`/`reference.py` and add a direct
  transcript test that asserts the exact offset sequence (not just that a stop happened).

- **A-3. Transactional phase-commit semantics.** [done] Pin the commit boundary: a phase's
  segment transforms commit together (one transcript epoch), so a verifier can see the
  basis jump only at segment boundaries. *Acceptance:* a transcript test asserting the
  per-phase commit structure.

### Track B — Stage 3: Seysen panel reference

- **B-1. Reference Seysen panel.** Transcribe the recursive Seysen panel (two-column
  Seysen leaf + rounded-inverse cross-panel) in exact `Fraction`, threaded through
  `lll_reduce` as a panel selector. *Acceptance:* matches FastInt64 Seysen output on the
  Seysen q-ary corpus, certified by `verify_reduction`.

- **B-2. Seysen postcondition (spec-first).** Define the exact Seysen-reduced
  postcondition and add `verify_reduction(mode="seysen")` + a corruption test that the
  verifier rejects a deliberately non-Seysen result. *This is spec work before code;
  record the definition in `docs/spec/` first.*

### Track C — Stage 4: DeepLLL reference

- **C-1. Reference DeepLLL.** Transcribe `deeplll_reduce`'s implemented depth predicate
  (`(i < depth || i >= k-depth)`) and adjacent-swap deep insertion. *Acceptance:* matches
  FastInt64 DeepLLL on the DeepLLL corpus; a transcript test pins the local depth-selection
  trajectory.

- **C-2. DeepLLL postcondition.** `verify_reduction(mode="deeplll")` postcondition +
  corruption rejection.

### Track D — Stage 5: BKZ reference

- **D-1. Progressive-beta schedule.** Transcribe `_progressive_betas` and add a direct
  schedule test (the implementation map flags this as untested).

- **D-2. Exact pruned enumeration.** The largest item; exact GH-radius + pruned
  enumeration is **not tick-sized** and needs its own `writing-plans` pass. Treat this
  as a *gate*, not a tick: before D-2, write a dedicated Stage-5 plan (spec + plan),
  review it, then implement. Do not improvise enumeration arithmetic.

### Track E — Engineering compatibility / precision polish

- **E-1. `verify_reduction` mode registry.** Reject unknown modes explicitly and document
  which modes have postconditions today (only `lll`, until B-2/C-2 land). *Acceptance:*
  an unknown `mode` raises `ValueError`, and the registry table is documented.

- **E-2. `get_profile` exactness for all input kinds.** `[done]` — Sage `is_upper`,
  Sage matrix, and object-dtype paths all exact (commits `ee6d99a`, `5a89354`, `7544538`).

- **E-3. Exact rational `delta`.** `[done]` — `delta_to_fraction` preserves exact
  `Fraction`/Sage `Rational` (commit `18a7401`).

- **E-4. CLI reference-engine surface.** `[done]` — reference engine rejects unsupported
  segmentation/Seysen/BKZ/anim/logfile flags and reports exact profile (commits
  `e02c629`, `f9d7687`, `2f049cc`, `9af0d7d`, `4c27bb7`). Revisit only to *enable* a flag
  once its stage lands.

- **E-5. `cores > 1` determinism.** Pin whether FastInt64 multi-core output is
  deterministic on a fixed corpus; document the answer (it is expected to be
  deterministic given the commit order, but this is currently untested — see the
  implementation map's "no deterministic transcript test").

- **E-6. Exact `stats` consumers.** Ensure `rhf`/`slope`/`potential`/`gaussian_heuristic`
  consume the exact `get_profile` output (`0.5*log2(r_i)` values) without regressing.
  *Acceptance:* a test running these on an arbitrary-precision basis.

### Track F — FastInt64 verification gaps (from the implementation map)

The remaining "coverage gaps that gate a reference implementation" are FastInt64-engine
*test* gaps, not reference-engine gaps. Pick them off one at a time, lowest-risk first:

- **F-1. Transactional checked-int64 overflow tests** for column addition and matrix
  product shapes.
- **F-2. Explicit normal-vs-partial (`KeyboardInterrupt`) completion status** test.
- **F-3. Definition-level classical size-reduction postcondition** test.
- **F-4. Direct offset / reduction-tree / front-tour / progressive-beta schedule** tests
  (overlaps A-2/D-1; do the reference-side versions first).

## 6. Per-tick protocol

1. Consult this document; take the first open item.
2. TDD: write the failing test, watch it fail for the right reason, write minimal code,
   watch it pass.
3. Run **both** suites: `make test` and `make test-sage` (preceded by the `build_ext`
   step the Makefile already performs). A green single-file run is not a green suite.
4. Commit the item independently, message ending with the Co-Authored-By line.
   Never stage `t.py`.
5. Report in Chinese: item done, test counts, commit hash, next item.
6. If an item needs design beyond its one-liner (D-2, B-2), **stop and write a
   spec/plan first** rather than improvising — that is the anti-drift behavior, not a
   stall.

## 7. Acceptance criteria (shared "done" test)

An item is done only when, in this session:

- a test for it exists and was watched failing before the code existed;
- the final `make test` and `make test-sage` runs passed and were read;
- the change is committed independently (its own commit, message ends with the
  attribution line);
- no constraint in §3 was violated and nothing in §9 was touched.

## 8. Definition of done for the loop + stop conditions

The loop is done when Track A–E are complete and the Stage-5 plan (D-2) has landed,
or earlier if every open item is blocked on an external decision. The loop runs until
an explicit **停** from the user; it never stops itself just because a stage finished
— the next stage is already specified above.

## 9. Explicit out-of-scope / do-not-drift-into

- Do **not** change the FastInt64 engine's arithmetic to "fix" int64 overflow.
- Do **not** route any `engine=` to Sage/fplll as the reduction engine.
- Do **not** introduce parallelism into the exact/reference engine.
- Do **not** add dependencies beyond Python stdlib + NumPy (input conversion) + Sage
  (isolated accelerator). No FLINT/GMP/NTL/MPFR selection before correctness evidence
  (ADR-0005).
- Do **not** alter the frozen M0 baseline expectations (iteration/quality values) without
  a spec version bump.
- Do **not** re-create the cron or call ScheduleWakeup; the loop is already scheduled.
- Do **not** commit `t.py`.
