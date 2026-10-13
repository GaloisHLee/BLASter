# BLASter arithmetic model (M2)

Version: 0.2 review candidate after the 2026-08-13 upstream sync. This is an
architecture contract, not an implementation.

The purpose of the arithmetic layer is to preserve the BLASter algorithm while
making integer overflow and uncertain real decisions explicit. It does not turn
Flatter, fpylll, or fplll into arithmetic backends, and it does not select a
final multiprecision library.

## Design principles

1. `B0`, `B`, and every transformation are exact state. QR/GSO and derived
   metrics are disposable approximate state.
2. `B0*U=B` is maintained by integer operations and checked independently.
3. A real decision is committed only when its comparison or rounding result is
   certified at the active precision.
4. Overflow or insufficient precision never triggers a different reduction
   algorithm. It triggers rollback, arithmetic escalation, and replay of the
   same BLASter phase, or a typed failure.
5. The reducer does not certify itself. An independent verifier consumes its
   result as untrusted data.

"Exact state" is an architectural role. FastInt64 preserves the current legacy
kernel and therefore cannot guarantee that role after an unchecked overflow; it
is explicitly reported as unverified rather than weakening the controlled
strategies' contract.

## Operation inventory

### Exact operations

These operations have integer mathematical semantics in every conforming
strategy:

| Operation | Objects | Required result |
|---|---|---|
| Parse and orient a basis | input, `B0` | exact integer entries and dimensions |
| Exact rank precheck | `B0` | rank over `Z`/`Q`, not a float estimate |
| Store current state | `B0`, `B`, `U` | no silent narrowing |
| Create identities/permutations | local `V`, panel `W`, `U` | exact unimodular matrices |
| Column addition | `b_j <- b_j+z*b_i` | exact integer multiply-add |
| Column swap/insertion | `B`, `U`, local transforms | exact permutation |
| Store enumeration coefficients | integer solution vector | exact integers after certified rounding |
| Compose local transforms | `U[:,S] <- U[:,S]*V` | exact matrix product |
| Compose panel transforms | `U <- U*W` | exact matrix product |
| Update the basis | `B[:,S] <- B[:,S]*V`, `B <- B*W` | exact matrix product |
| Reconstruct output | `B_check <- B0*U` | exact matrix product |
| Check transformation relation | `B_check == B` | exact equality |
| Check unimodularity | `det(U)` | exact determinant equal to `+1` or `-1` |
| Check lattice equivalence | input/output/transform | exact relation plus unimodularity |

For a row-oriented API the exact relation is `T*A0=A`, with `T=transpose(U)`.
No floating determinant is admissible as verification evidence.

### Controlled approximate operations

These use real arithmetic, but some of them produce control-flow decisions or
exact integer coefficients and therefore require uncertainty handling:

| Operation | Use | Control requirement |
|---|---|---|
| QR/GSO of exact `B` | construct `R` | error-bounded or repeatable at higher precision |
| Givens update after a swap | restore triangular form | error-bounded; discarded on replay |
| LLL quotient `-R[i,j]/R[i,i]` | size coefficient | certified `nearest_core` |
| Local Lovasz comparison | swap decision | certified comparison including equality rule |
| Weak-LLL predicate | global stop | certified strict comparison |
| Size-reduction predicate | postcondition | certified non-strict comparison |
| DeepLLL projected norms | insertion decision | certified sum and comparison |
| Babai target propagation | panel coefficient | error-bounded multiply/add and certified `nearest_panel` |
| Seysen triangular inverse/solve | panel coefficient | residual/error bound and certified `nearest_panel` |
| Floating panel products | update approximate `R` | error-bounded; exact `B` remains authoritative |
| GSO ratios and squared norms | enumeration input | error-bounded |
| GH/log/gamma calculation | enumeration radius | error-bounded comparison against candidates |
| Pruning-bound comparison | enumeration traversal | certified comparison; replay on escalation |
| Enumeration center | coefficient order | certified `nearest_core` |
| Candidate norm comparison | best-vector selection | certified comparison with fixed tie behavior |
| Profile, RHF, slope, potential | diagnostics only | record precision; never establish correctness |

Diagnostic metrics may be returned with an explicit error estimate or marked
non-certified. They do not by themselves force precision escalation unless the
caller requests certified diagnostics.

## Layered interfaces

The names below describe responsibilities, not a required programming language.

### Exact integer domain

```text
interface IntegerDomain:
    scalar(value) -> Integer
    matrix(rows, columns, values) -> IntegerMatrix
    identity(n) -> IntegerMatrix
    addmul_column(M, source, target, coefficient) -> IntegerMatrix
    swap_columns(M, i, j) -> IntegerMatrix
    matmul(A, B) -> IntegerMatrix
    compose_in_place(A, B) -> IntegerMatrix
    determinant(A) -> Integer
    equal(A, B) -> bool
    fits_int64(value_or_matrix) -> bool
```

Each mutating operation is atomic with respect to an arithmetic checkpoint: it
either commits a complete exact result or leaves the prior matrix unchanged.
The interface never exposes wraparound arithmetic.

### Controlled real context

```text
interface RealContext:
    precision_bits -> integer
    from_integer_matrix(B) -> RealMatrix
    qr_r(B) -> RealMatrix
    matmul(A, B) -> RealMatrix
    triangular_solve(A, B) -> RealMatrix
    dot/norm/sqrt/log/exp/log_gamma(...)
    compare(x, y) -> LESS | EQUAL | GREATER | UNDECIDED
    nearest_core(x) -> Integer | UNDECIDED
    nearest_panel(x) -> Integer | UNDECIDED
    error_bound(value) -> bound
```

`EQUAL` is returned only when equality is proved, for example from an exact
expression retained by the context. An interval containing both sides returns
`UNDECIDED`, not `EQUAL`. Matrix inverse is not an interface primitive: Seysen
uses a triangular solve with a residual/error bound, even though the current
implementation explicitly forms an inverse.

### Matrix operation layer

```text
interface ArithmeticSession:
    integer_domain
    real_context
    checkpoint(exact_state, scheduler_state) -> CheckpointId
    commit(operation)
    rollback(CheckpointId)
    escalate_real(reason, CheckpointId)
    promote_integer(reason, CheckpointId)
    event_log
```

This layer owns optimized matrix kernels, range proofs, temporary storage, and
transaction boundaries. Reduction code asks for mathematical operations; it
does not call Eigen, BLAS, MPFR, or an arbitrary-integer library directly.

## Exact and approximate state

An exact checkpoint contains:

```text
B0, B, U
mode and all public parameters
offset or BKZ (beta stage, front, tours_done)
phase identifier and iteration counters
committed decision-transcript prefix
```

An approximate epoch contains:

```text
checkpoint identifier
real backend and precision
R and all derived GSO/panel/enumeration values
provisional local transformations not yet committed
```

Approximate state may never outlive or be attached to a different exact
checkpoint. A precision escalation discards the entire approximate epoch,
restores the checkpoint's scheduler state, converts the exact `B` again, and
recomputes QR before replay. Converting the old binary64 `R` to a wider type is
forbidden because it preserves error without restoring information.

Required checkpoint boundaries are:

- before each segmented LLL local phase;
- after local transforms commit and before classical/Seysen panel construction;
- before each segmented BKZ local phase, including all local enumerations; and
- after the panel commits and before the embedded segmented LLL call.

All transformations computed concurrently in one segmented phase are
provisional until every segment succeeds. One segment requesting escalation or
overflow causes the entire phase to roll back and replay, preventing a
precision-dependent mixture of segment results.

## Parameter representation

Algorithm parameters that participate in decisions are immutable session data.
For M0 compatibility, `delta` is converted once to IEEE-754 binary64 exactly as
the current API does, and its bit pattern (equivalently, exact integer ratio) is
recorded. Checked and multiprecision real contexts import that exact value; they
do not reinterpret the display text `0.99` as the rational `99/100`.

Integer parameters and rational constants such as `1023/1024` remain exact until
conversion into the active real context. Transcendental results such as
`log_gamma` carry the context's error bound. A future API that accepts an exact
rational `delta` may be useful, but it requires an explicit API/algorithm-profile
decision because it can change comparisons relative to the frozen baseline.

## Rounding contract

The arithmetic layer implements the two operators from the algorithm
specification:

```text
nearest_core:  nearest integer, exact tie away from zero
nearest_panel: nearest integer, exact tie to even
```

For a real enclosure `X`, rounding is certified only when every value in `X`
maps to the same integer under the relevant rule. If `X` intersects a boundary
`k+1/2`, the result is `UNDECIDED` and precision increases. If an exact
half-integer is proved, the named tie rule is applied directly. An integer that
does not fit the current exact integer domain raises an integer-range event; it
is never narrowed by a cast.

The site-specific operators preserve the current algorithm. Unifying their tie
rules in a future version would be an algorithm decision, not an arithmetic
refactor.

## Certified comparisons and precision escalation

Every control expression is reduced to the sign of an error-bounded quantity.
For example, local Lovasz uses

```text
D = R[k-1,k]^2 + R[k,k]^2 - delta*R[k-1,k-1]^2.
```

The non-strict local condition is accepted when `D > 0` or `D = 0` is proved;
it is rejected when `D < 0`. An enclosure spanning zero is undecided. The global
weak predicate similarly uses its specified strict sign, so proved equality is
not reduced.

Controlled strategies use this precision ladder by default:

```text
p0 = 53 bits
p_next = min(2*p_current, max_real_bits)
max_real_bits = 4096 bits
```

Both values are configuration and result metadata. An implementation may skip
directly to a larger precision when its error estimate predicts that doubling
cannot decide the expression. It may not exceed `max_real_bits` silently.

On `UNDECIDED`:

1. record the operation, expression class, checkpoint, and current precision;
2. roll back provisional integer and scheduler changes;
3. increase precision;
4. reconstruct real state from the exact checkpoint basis; and
5. replay the same BLASter phase with the same algorithm parameters and fixed
   traversal order.

If the decision remains unproved at 4096 bits, the operation returns
`PrecisionExhausted`. The top-level result has a failed or partial status and no
reducedness claim. It must not guess, silently use the current midpoint, call an
external reducer, or treat an interval overlap as equality.

Enumeration is subject to the same rule. Escalation discards the current search
tree, radius updates, and candidate; it recomputes GSO and restarts the local
enumeration in its specified order.

## Signed-int64 overflow control

The signed range is `[-2^63, 2^63-1]`. Input range checking is necessary but not
sufficient because transformation coefficients and matrix products can grow.

### Scalar and column operations

Checked operations evaluate products and sums in a wider exact scratch domain,
verify the final value against the signed range, and commit only after every
entry succeeds. This applies to:

- `z * U[:,i] + U[:,j]` in local column addition;
- the corresponding exact basis-column update;
- rounded coefficients before storing them; and
- negation of `-2^63`, which is itself out of signed range.

Compiler overflow builtins or 128-bit integers may implement fast checks when
their own range is proved sufficient. Arbitrary-precision scratch is the final
checker; signed C++ overflow is never used as a detection mechanism.

### Matrix multiplication preflight

For `C=A*B` with inner dimension `k`, a cheap sufficient global bound is

```text
k * max_abs(A) * max_abs(B) <= 2^63 - 1.
```

A tighter sufficient bound checks each output:

```text
sum(t=0..k-1, abs(A[i,t]) * abs(B[t,j])) <= 2^63 - 1.
```

Bounds are themselves computed without overflow. If a sufficient bound passes,
every product and partial sum is in range and an optimized int64 kernel may run.
If the bound does not pass, overflow is not yet proved: cancellation may keep
the final result in range. The checked path therefore computes each dot product
in an exact scratch accumulator and range-checks the final entry.

In-place products allocate a complete temporary result. Only after every entry
fits is the destination replaced. On failure, `IntegerOverflow` reports the
operation kind, matrix shapes, offending output coordinate/value or bound,
checkpoint, and configured promotion policy. No destination is partially
updated.

## Arithmetic strategies

### FastInt64

Purpose: preserve and isolate the current high-speed proof-of-concept behavior.

- Exact-state storage type: signed int64.
- Integer kernels: unchecked current C++/Eigen operations.
- Real context: fixed binary64.
- Real escalation: none.
- Input: strictly integer and in signed range.
- Failure model: intermediate overflow is not reliably detectable and C++
  signed overflow is outside the mathematical contract; near-boundary real
  decisions use the binary64 result.
- Result status: `fast_unverified` until an independent exact verifier accepts
  it. Verification can detect many corruptions but cannot retroactively prove
  the execution had no undefined behavior.

FastInt64 is a compatibility/performance strategy, not the default evidence for
correctness-sensitive runs.

### CheckedInt64

Purpose: retain compact int64 storage and fast kernels when a proof shows they
are safe.

- Exact-state storage type: signed int64.
- Integer kernels: preflighted fast path or transactional exact-scratch path.
- Real context: controlled precision beginning at 53 bits.
- Integer overflow: typed `IntegerOverflow` before commit.
- Real ambiguity: rollback and precision escalation up to 4096 bits.
- Failure policy: either return the typed failure or, when explicitly configured
  for promotion, replay from the checkpoint with Multiprecision integers.
- Result status: eligible for independent verification and a `verified` result.

CheckedInt64 never falls back to unchecked multiplication after an inconclusive
bound.

### Multiprecision

Purpose: execute the same BLASter algorithm without a fixed integer range.

- Exact-state storage type: arbitrary-precision signed integers.
- Integer kernels: exact arbitrary-precision operations.
- Real context: controlled arbitrary precision with the same 53-to-4096-bit
  default ladder.
- Integer overflow: impossible in the mathematical model; allocation/resource
  failure remains possible.
- Real ambiguity: rollback and precision escalation exactly as in CheckedInt64.
- Failure model: `PrecisionExhausted`, `ResourceLimit`, `Interrupted`, or
  `VerificationFailed`; no external-algorithm substitution.
- Result status: eligible for independent verification and a `verified` result.

Multiprecision is a design target only in M2. No multiprecision core is
implemented in this milestone.

## Strategy transitions

```text
                         real ambiguity
                    +----------------------+
                    | increase real p and  |
                    | replay exact epoch   |
                    v                      |
CheckedInt64 ------------------------------+
     |
     | proved/predicted integer overflow and promotion enabled
     v
Multiprecision ----------------------------+

FastInt64 -- explicit restart --> CheckedInt64 or Multiprecision
```

Rules:

- FastInt64 cannot be promoted live after a suspected overflow because its last
  state may already be corrupt. Transition requires an explicit restart from
  `B0` or from a checkpoint created before any unchecked operation.
- CheckedInt64 promotes transactionally from the checkpoint preceding the
  failed operation. The exact int64 checkpoint values are losslessly converted,
  QR is recomputed, and the whole phase is replayed.
- Increasing real precision does not require promoting integer storage.
- Multiprecision never downgrades within a run. A result may be exported to
  int64 only after a complete range proof; export is not a strategy transition.
- No transition invokes Flatter, fpylll, or fplll.

## Algorithm identity and decision trajectories

**Specification equivalence** means two runs use the same version of
[`blaster-algorithm.md`](blaster-algorithm.md), parameters, rounding operators,
partition/traversal order, and output contract. Both return independently
verified, lattice-equivalent results satisfying the mode's specified
postcondition.

**Trajectory equivalence** is stronger: the runs have the same ordered local
coefficients, swaps, insertions, segment transforms, enumeration candidate
updates, and stop decisions. It is represented by a decision transcript, not by
equal final RHF.

The policy is:

- Arithmetic backends may not deliberately choose different predicates,
  rounding rules, traversal orders, or reduction algorithms.
- Escalation may revise a provisional low-precision decision before commit. That
  is required behavior, not an algorithm change.
- A controlled run is allowed to differ from FastInt64 when the latter made an
  uncertified binary64 decision or suffered overflow.
- CheckedInt64 and Multiprecision should be trajectory-equivalent when both can
  certify every decision from the same checkpoint. Different escalation counts
  are allowed and recorded.
- If a unique specified decision cannot be certified, the run fails instead of
  choosing a backend-dependent trajectory.

Therefore cross-backend byte-for-byte output is not the correctness definition;
the algorithm specification and independent postconditions are. A dedicated
compatibility test profile may additionally demand trajectory equality on the
M0 corpus.

## Result and event metadata

Every non-legacy result object must record:

| Field | Meaning |
|---|---|
| `algorithm_spec` | specification version/hash |
| `parameters` | all effective values, including internal DeepLLL-4 policy |
| `integer_strategy` | FastInt64, CheckedInt64, or Multiprecision |
| `integer_backend` | concrete library/kernel and version |
| `real_backend` | concrete real/ball implementation and version |
| `initial_precision_bits` | first real precision |
| `maximum_precision_bits` | configured ceiling |
| `final_precision_bits` | precision of the final committed decision epoch |
| `escalations` | ordered checkpoint, operation, reason, old/new precision or integer strategy |
| `overflow_preflights` | fast-path passes, exact-scratch checks, and promotion count |
| `decision_transcript_digest` | digest of the ordered committed decisions |
| `completion_status` | normal, interrupted, timeout, iteration limit, stagnation, or failed |
| `verification` | verifier identity/version, checks, precision, and outcome |
| `diagnostics_precision` | precision/error status for profile, RHF, slope, and potential |

FastInt64's legacy tuple API can remain for compatibility. New checked APIs must
not discard this metadata.

## Independent verification

The verifier is a separate component with no access to reducer scratch buffers
or trust in its claims. Preferably it uses a different implementation or binding
for exact arithmetic. Given `B0`, returned `B`, `U`, mode, and parameters, it:

1. parses all entries into arbitrary-precision integers;
2. checks dimensions and exact rank;
3. recomputes `B0*U` and compares it with `B`;
4. computes `det(U)` exactly and requires `abs(det(U))=1`;
5. recomputes QR/GSO independently at controlled precision;
6. checks only the postconditions promised by the mode; and
7. reports `UNDECIDED` and escalates its own precision independently of the
   reducer, up to its separately configured maximum.

A verifier failure returns `VerificationFailed` and preserves the result as
diagnostic data without labelling it valid. Flatter/fpylll output may be another
benchmark sample, but it is not the verifier's specification truth.

## Candidate implementation technologies

No final backend is selected in M2. Candidates are evaluated by responsibility:

| Candidate | Plausible responsibility | Advantages | Integration costs/limits |
|---|---|---|---|
| FLINT | exact integers and dense integer matrices (`fmpz`, matrix layer); possibly Arb-backed rigorous reals through the wider FLINT ecosystem | optimized small-inline/large-GMP integer representation, number-theory matrix operations, C ABI | ownership/threading/build integration; choosing its higher-level algorithms must not replace BLASter control flow |
| GMP | scalar arbitrary-precision integer foundation and exact scratch accumulators | mature low-level `mpz` arithmetic and precise control | no BLASter-specific matrix abstraction; more custom matrix/kernel and lifetime code |
| NTL | C++ arbitrary integers and integer matrices | RAII-style C++ surface and useful matrix types | adapter, allocation, threading, and dependency behavior require measurement; NTL reduction routines cannot substitute for BLASter |
| MPFR | controlled arbitrary-precision real scalars with explicit rounding | well-defined precision and rounding for QR/GSO building blocks | no integer or dense matrix layer; rigorous matrix error bounds and kernels still need design |

FLINT currently promotes large `fmpz` values to GMP-backed storage while keeping
small values inline; this makes it a strong matrix-layer candidate rather than a
different reduction algorithm. MPFR remains a real-layer candidate regardless
of whether exact integers come from FLINT, GMP, or NTL. Arb-style ball arithmetic
is also relevant for certified comparisons, but adopting it is a later backend
decision.

Selection evidence required after the Sage reference implementation includes:

- exact and checked matrix-product throughput on representative BLASter shapes;
- conversion and checkpoint/replay cost;
- support for deterministic threading and fixed traversal order;
- QR/GSO error-bound feasibility and precision-escalation ergonomics;
- cross-platform build and packaging cost in the Sage 10.6 environment; and
- independent-verifier diversity.

Sources for candidate scope are the
[FLINT repository and documentation](https://github.com/flintlib/flint),
[GMP manual](https://gmplib.org/manual/index.html),
[NTL documentation](https://libntl.org/doc/tour.html), and
[MPFR manual](https://www.mpfr.org/mpfr-current/mpfr.html). Context7 resolution
for this review identified `/flintlib/flint`; its current source confirms the
small-inline/large-GMP `fmpz` dispatch described above.

## Typed failures

The arithmetic boundary defines these failures:

| Failure | Meaning | State returned |
|---|---|---|
| `IntegerOverflow` | CheckedInt64 result cannot fit and promotion is disabled | last exact checkpoint |
| `PrecisionExhausted` | a required decision remains uncertified at maximum precision | last exact checkpoint plus undecided expression metadata |
| `ResourceLimit` | memory or configured arithmetic resource ceiling reached | last exact checkpoint when recoverable |
| `Interrupted` | user interruption | last committed exact checkpoint |
| `VerificationFailed` | independent structural or reduction check failed | untrusted reducer output plus verifier evidence |
| `BackendFailure` | backend invariant or operation failed | last exact checkpoint and backend diagnostics |

No failure is translated into a normal reduced result. Partial exact state may be
returned for diagnosis or an explicit resume operation, always with its status.

## Non-goals for M2

- implementing a multiprecision C++ core;
- changing current reducer results, defaults, or local formulas;
- optimizing reduction performance;
- adding another reduction algorithm;
- runtime routing among BLASter, Flatter, and fpylll;
- using an external reducer as BLASter's specification oracle; or
- refactoring the current core before the reference specification is reviewed.

The implementation order and review boundaries are recorded in the ADRs under
[`docs/adr/`](../adr/).
