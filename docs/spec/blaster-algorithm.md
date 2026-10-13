# BLASter algorithm specification (M1)

Version: 0.2 review candidate, synced with upstream `68225ec` on 2026-08-13.

This document specifies BLASter as an algorithm, independently of Python,
NumPy, Cython, Eigen, machine integers, and any particular multiprecision
library. It is detailed enough to implement a slow reference reducer. Arithmetic
requirements are defined separately in
[`arithmetic-model.md`](arithmetic-model.md).

## Status vocabulary

- **Specified** describes the algorithm that a conforming BLASter
  implementation must expose.
- **Current behavior** records a choice made by this worktree that affects the
  trajectory or result.
- **Known limitation** records a property that is neither proved nor certified.

The specified control flow intentionally follows current behavior, including
unusual choices. A later algorithm change requires a new specification version
and must not be disguised as an arithmetic-backend change.

## Mathematical model

### Basis orientation

The canonical representation stores basis vectors as columns. For rank `n` in
ambient dimension `m >= n`:

```text
B0 in Z^(m x n)     original full-column-rank basis
U  in GL(n, Z)      accumulated unimodular transformation
B  in Z^(m x n)     current basis
B = B0 * U          exact structural invariant
```

The public row-oriented API accepts `A0 = transpose(B0)`. It returns
`A = transpose(B)` and `T = transpose(U)`, so its invariant is
`T * A0 = A`. Orientation conversion is an API boundary only; the algorithm
below always uses columns.

### QR state

A thin QR decomposition of the current basis is

```text
B = Q * R,
```

where the columns of `Q` are orthonormal and `R` is an `n x n` upper-triangular
real matrix. BLASter stores only `R`. The signs of diagonal entries are not
normalized. Predicates therefore use `abs(R[i,i])` where sign is irrelevant.

An `R` value is valid only for the exact basis state from which it was computed,
or after a right upper-triangular panel transformation that was applied to both
`R` and `B`. Any permutation, deep insertion, or independently reduced segment
invalidates the global `R` and requires QR recomputation.

### Two specified nearest-integer operators

Current behavior has two tie rules, so the algorithm names both explicitly:

- `nearest_core(x)` returns the nearest integer and resolves exact half-integers
  away from zero. It is used by local LLL and SVP enumeration.
- `nearest_panel(x)` returns the nearest integer and resolves exact
  half-integers to the even integer. It is used by classical panel reduction,
  Babai reduction, and Seysen panel reduction.

This distinction is deliberate compatibility behavior, not a recommendation for
new algorithms. A backend must implement both operators identically; choosing a
host language's unnamed `round` operation is not conforming.

## Reduction predicates

For adjacent positions `i` and `i+1`, define

```text
u = R[i,i], v = R[i,i+1], w = R[i+1,i+1].
```

Following upstream `68225ec`, `R` is **weak-LLL(delta)** exactly when, for every
adjacent pair,

```text
v^2 + w^2 > delta * u^2.
```

There is no modulo reduction of `v` in this predicate. The strict inequality is
current behavior: equality is reported as not reduced.
`R` is **size-reduced** when

```text
abs(R[i,j]) <= abs(R[i,i]) / 2 for every 0 <= i < j < n.
```

`R` is **LLL(delta)** in BLASter's checker when it is both weak-LLL and
size-reduced.

Local LLL accepts the Lovasz condition with a non-strict comparison:

```text
delta * R[k-1,k-1]^2 <= R[k-1,k]^2 + R[k,k]^2.
```

Consequently, an exact equality can be accepted by local LLL and rejected by the
global weak-LLL checker. This is a specified description of the current
comparison-boundary mismatch, not a proof that the two predicates are
equivalent.

## Segments

For rank `n`, offset `o`, and segment width `s`, the segment set is

```text
[o, min(o+s,n)), [o+s, min(o+2s,n)), ...
```

Empty segments are omitted. Positions before `o` are untouched in that local
phase; segments do not wrap around. Since segments are disjoint, their local
transformations may be computed concurrently. They are committed to `B` and `U`
in increasing segment order.

The segmented LLL outer loop initializes `offset = 0` and toggles before each
local phase:

```text
offset = floor(s/2) if offset == 0 else 0.
```

Thus the first phase starts at `floor(s/2)`, the second at zero, and subsequent
phases alternate. This half-segment overlap is how transformations cross the
boundaries of the previous partition.

## Elementary local operations

### Integer column addition

For `i < j` and integer `z`, `ALTER(i,j,z)` performs

```text
R[0:i+1,j] <- R[0:i+1,j] + z * R[0:i+1,i]
V[:,j]     <- V[:,j]     + z * V[:,i].
```

Here `V` is the local transformation, initialized to identity. This represents
replacing local basis column `j` by `b_j + z*b_i`.

### Adjacent swap

`SWAP(k,k+1)` swaps adjacent basis columns and restores triangular form with a
Givens rotation. Given the old values

```text
a = R[k,k], c0 = R[k,k+1], s0 = R[k+1,k+1],
rho = sqrt(c0^2 + s0^2), c = c0/rho, s = s0/rho,
```

it sets

```text
R[k,k]       = rho
R[k,k+1]     = c * a
R[k+1,k+1]   = s * a
R[k,k+2:]    = c * old(R[k,k+2:]) + s * old(R[k+1,k+2:])
R[k+1,k+2:]  = s * old(R[k,k+2:]) - c * old(R[k+1,k+2:]),
```

then swaps the entries above rows `k` in columns `k,k+1`, and swaps columns
`k,k+1` of `V`. A zero `rho` is outside the valid full-rank QR precondition.

## Local LLL

Input is a square diagonal block `R_S` for one segment, a value
`0.25 < delta < 1`, and no state outside the segment. Output is a unimodular
local transformation `V`. Only `B[:,S]` and `U[:,S]` may be changed when `V` is
committed.

```text
LOCAL_LLL(R_S, delta):
    V <- identity(width(S))
    k <- 1
    while k < width(S):
        for i <- k-1 down to 0:
            z <- nearest_core(-R_S[i,k] / R_S[i,i])
            ALTER(i, k, z)

        if delta * R_S[k-1,k-1]^2
                <= R_S[k-1,k]^2 + R_S[k,k]^2:
            k <- k + 1
        else:
            SWAP(k-1, k)
            if k > 1:
                k <- k - 1
    return V
```

The local routine mutates its private `R_S` to make decisions. The global
implementation need not splice this approximate block back into `R`; it applies
`V` to exact `B` and `U`, then recomputes global QR.

## Local DeepLLL

The current depth rule is not simply "insert at most `depth` positions back".
For candidate insertion index `i < k`, it permits `i` when

```text
i < depth or i >= k - depth.
```

This includes the first `depth` positions even when they are far from `k`, plus
the last `depth` positions before `k`.

```text
DEEPLLL_PREFIX_IN_PLACE(R_S, V, delta, depth, limit):
    LLL_PREFIX_IN_PLACE(R_S, V, delta, limit)
    k <- 1
    while k < limit:
        for i <- k-1 down to 0:
            z <- nearest_core(-R_S[i,k] / R_S[i,i])
            ALTER(i, k, z)

        projected <- sum(R_S[r,k]^2 for r in 0..k)
        inserted <- false
        for i <- 0 to k-1:
            permitted <- (i < depth) or (i >= k-depth)
            if permitted and projected <= delta * R_S[i,i]^2:
                while k > i:
                    k <- k - 1
                    SWAP(k, k+1)
                if k == 0:
                    k <- 1
                inserted <- true
                break
            projected <- projected - R_S[i,k]^2
        if not inserted:
            k <- k + 1

LOCAL_DEEPLLL(R_S, delta, depth):
    V <- identity(width(R_S))
    DEEPLLL_PREFIX_IN_PLACE(R_S, V, delta, depth, width(R_S))
    return V
```

`LLL_PREFIX_IN_PLACE` is the local LLL loop restricted to `k < limit`; it
composes into the supplied `V` rather than initializing it. Thus DeepLLL first
runs local LLL using the same `V`, and BKZ can invoke it repeatedly without
losing earlier transformations. Segment mode commits the final `V` in the same
way as ordinary local LLL.

## Classical panel size reduction

Classical reduction constructs a unit upper-triangular transformation `W` with a
balanced divide-and-conquer tree. The following recursive pseudocode is
algebraically equivalent to the current non-recursive range scheduler under
exact arithmetic.

`BABAI(R,T)` treats each column of `T` as a target:

```text
BABAI(R, T):
    C <- zero_integer_matrix(rows(R), columns(T))
    for j <- rows(R)-1 down to 0:
        C[j,:] <- -nearest_panel(T[j,:] / R[j,j])
        T[0:j+1,:] <- T[0:j+1,:] + R[0:j+1,j] * C[j,:]
    return (T, C)
```

`SIZE_PANEL(R)` is:

```text
SIZE_PANEL(R):
    if size(R) == 1:
        return (R, identity(1))
    split R at m into blocks R11, R12, R22
    (S11, W11) <- SIZE_PANEL(R11)
    (S22, W22) <- SIZE_PANEL(R22)
    T <- R12 * W22
    (S12, C) <- BABAI(S11, T)
    W12 <- W11 * C
    S <- [[S11, S12], [0, S22]]
    W <- [[W11, W12], [0, W22]]
    return (S, W)
```

For a two-column leaf this gives
`W[0,1] = -nearest_panel(R[0,1]/R[0,0])`. The intended postcondition is
`S = R*W` and size reduction. Binary64 scheduling can change rounding near a
boundary, so the current implementation does not certify that postcondition.

## Seysen panel reduction

BLASter's `use_seysen` mode has the same recursive tree and unit
upper-triangular form, but computes each cross panel by a rounded solve rather
than Babai's descending nearest-plane pass:

```text
SEYSEN_PANEL(R):
    if size(R) == 1:
        return (R, identity(1))
    split R at m into blocks R11, R12, R22
    (S11, W11) <- SEYSEN_PANEL(R11)
    (S22, W22) <- SEYSEN_PANEL(R22)
    T <- R12 * W22
    C <- nearest_panel(-inverse(S11) * T) entrywise
    S12 <- T + S11 * C
    W12 <- W11 * C
    S <- [[S11, S12], [0, S22]]
    W <- [[W11, W12], [0, W22]]
    return (S, W)
```

The name identifies the current recursive Seysen-style transformation. The
implementation does not evaluate a separate global Seysen measure or certify a
standard Seysen-reduced postcondition. Global termination checks weak LLL only,
not size reduction or a Seysen condition.

The current implementation reuses one identity-initialized panel scratch matrix
between outer iterations. Its range schedule overwrites every strict
upper-triangular block and relies on the lower triangle remaining zero and the
diagonal remaining one. A reference implementation may allocate a fresh `W` on
each call; that does not change the specified algorithm.

## Segmented LLL state machine

`LOCAL_MODE` is `LOCAL_LLL` when `depth=0`, otherwise `LOCAL_DEEPLLL`.

```text
SEGMENTED_LLL(B0, B, U, segment_size, delta, depth, panel_mode):
    offset <- 0
    repeat:
        R0 <- QR_R(B)                                      # QR checkpoint A
        offset <- floor(segment_size/2) if offset == 0 else 0

        for each S in SEGMENTS(rank(B), offset, segment_size), concurrently:
            V_S <- LOCAL_MODE(copy(R0[S,S]), delta, depth)
        for each S in increasing order:
            B[:,S] <- B[:,S] * V_S
            U[:,S] <- U[:,S] * V_S

        R1 <- QR_R(B)                                      # QR checkpoint B
        if panel_mode == CLASSICAL:
            (R2, W) <- SIZE_PANEL(R1)
        else:
            (R2, W) <- SEYSEN_PANEL(R1)
        B <- B * W
        U <- U * W

        stopped <- WEAK_LLL(R2, delta)
    until stopped
    return (B, U)
```

There are exactly two global QR computations per outer iteration: immediately
before local segments and immediately after their exact transformations. There
is no third QR after the panel transform because `R2` is intended to equal
`R1*W` while remaining upper triangular. The next iteration recomputes QR from
the exact basis.

## Local SVP and BKZ tour

Enumeration supports at most 80 dimensions. For a local SVP window beginning at
`i` with width `w`, current behavior computes:

```text
log_volume_argument = sum(log(R[j,j]^2) for j in i..i+w-1)
gh2(w,l) = exp(2 * (log_gamma(w/2 + 1) + l) / w) / pi
factor = max(1.05, min(2.0, 1.0 + 4.0/w))
radius2 = min((1023/1024) * R[i,i]^2, gh2(w,log_volume_argument) * factor^2)
```

The helper's comment calls its second argument the logarithm of lattice volume,
but the caller supplies the logarithm of squared volume. The formula above,
rather than the comment, is the current trajectory definition. Correcting it is
an algorithm change requiring separate review.

The enumerator uses the versioned relative-pruning vector `P_w` stored in
`core/pruning_params.cpp`. Define

```text
mu[i,j] = R[i,j] / R[i,i] for i < j
r[i] = R[i,i]^2
center[i] = -sum(mu[i,j] * x[j] for j=i+1..w-1)
partial[i] = partial[i+1] + (center[i]-x[i])^2 * r[i]
```

It performs depth-first Schnorr-Euchner-style enumeration from level `w-1` down
to zero. At each level it visits `nearest_core(center[i])` first, then alternates
the integers on either side in increasing distance. On the all-zero higher-level
path it visits only the positive representative after zero, removing sign
symmetry. A node survives level `i` when
`partial[i] <= radius2 * P_w[i]`. At a leaf, the zero vector is rejected; a
nonzero vector with `partial[0] <= radius2` replaces the saved solution, including
on equality, and shrinks `radius2` to `partial[0]`. The current fixed-size engine
pads to dimension 80 with orthogonal virtual vectors longer than the radius;
their coefficients must remain zero. A slow reference implementation may omit
that padding while preserving the actual-dimension traversal and tie order.

The result is the best coefficient vector encountered in this pruned traversal,
or zero when none is found. A returned vector is inserted only if its last
nonzero coefficient is `+1` or `-1` and its last nonzero position `j` is greater
than zero. A `-1` vector is negated; then the current column `i+j` is replaced by
the integer combination and moved to `i` by adjacent swaps. Other candidates are
discarded. Whether or not insertion succeeds, DeepLLL-4 is run on the local
prefix ending at `min(i+w,N)`.

```text
LOCAL_BKZ_TOUR(R, delta, beta):
    V <- identity(N)
    DEEPLLL_PREFIX_IN_PLACE(R, V, delta, depth=4, limit=N)
    if beta <= 2:
        return V
    if beta > N:
        windows <- [ [i,N) for i in 0..N-2 ]
    else:
        windows <- [ [i,i+beta) for i in 0..N-beta ]
    for window in windows:
        candidate <- PRUNED_ENUMERATION(window)
        if candidate has an easy +/-1 insertion:
            insert it at the window front, updating R and V
        DEEPLLL_PREFIX_IN_PLACE(R, V, delta, depth=4,
                                limit=min(window_end,N))
    return V
```

When `beta > N` at a short final segment, the suffix windows are called an
HKZ-style tour in the implementation. Heuristic pruning and discarded difficult
insertions mean this is not a certified HKZ oracle.

## Segmented BKZ and progressive beta

Let `segment_size` here mean `bkz_size`. One segmented BKZ stage is:

```text
SEGMENTED_BKZ(B0, B, U, beta, tours, segment_size, ...):
    (B,U) <- SEGMENTED_LLL(..., depth=4, ...)
    tours_done <- 0
    front <- 0
    while tours_done < tours:
        R0 <- QR_R(B)
        offset <- front mod beta
        for each S in SEGMENTS(rank(B), offset, segment_size), concurrently:
            V_S <- LOCAL_BKZ_TOUR(copy(R0[S,S]), delta, beta)
        commit every V_S to B and U in increasing segment order

        R1 <- QR_R(B)
        (R2,W) <- selected panel reduction(R1)
        B <- B * W
        U <- U * W

        if front + beta > rank(B):
            front <- 0
            tours_done <- tours_done + 1
        else:
            front <- front + (segment_size - beta + 1)

        (B,U) <- SEGMENTED_LLL(..., depth=4, ...)
    return (B,U)
```

`front` does not directly select one global BKZ window. It selects the partition
offset through `front mod beta`; every segment after that offset receives a local
BKZ tour. The strict `front + beta > n` test means equality does not end a tour.
Because validation requires `segment_size >= beta`, the front increment is at
least one.

For target `beta` and progressive increment `p`, upstream behavior is:

```text
if beta < 40:
    [ ]                       # run DeepLLL-4, no BKZ tour
else if p >= beta:
    [beta]
else:
    start = 40 + ((beta - 40) mod p)
    [start, start+p, ..., beta]
```

Each generated beta receives one segmented tour; the target beta receives the
requested number of tours. Every stage performs its own initial DeepLLL-4
segmented reduction. Public input cannot combine `depth` and `beta`; the BKZ
path nevertheless uses its fixed internal depth of four.

The call described in source as a "final LLL reduction at the end" actually runs
after every segmented BKZ iteration, including iterations that have not ended a
tour. The pseudocode captures the executable behavior.

## Top-level algorithm

```text
BLASTER(B0, parameters):
    validate shape, full rank, and parameters
    B <- B0
    U <- identity(rank(B0))

    if beta is absent:
        return SEGMENTED_LLL(B0, B, U, lll_size, delta,
                             depth, panel_mode)

    if beta < 40:
        return SEGMENTED_LLL(B0, B, U, lll_size, delta, depth=4,
                             panel_mode)

    for beta_i in PROGRESSIVE_BETAS(beta, bkz_prog):
        stage_tours <- bkz_tours if beta_i == beta else 1
        (B,U) <- SEGMENTED_BKZ(B0, B, U, beta_i, stage_tours,
                               bkz_size, lll_size, delta,
                               panel_mode)
    return (B,U)
```

Current validation requires `m >= n`, exact full rank (modular GF(p), see
"Known limitations"), `0.25 < delta < 1`,
`lll_size >= 2`, `cores >= 1`, `depth >= 0`, and, for BKZ,
`2 <= beta <= min(n,80)`, positive tours, `bkz_size >= beta`, and
`1 <= bkz_prog <= beta`. Segment widths are capped at rank.

## Completion, interruption, and limits

Current ordinary/DeepLLL mode has no iteration bound, stagnation detector, or
timeout. BKZ has a finite requested tour count, but every embedded segmented LLL
call is likewise unbounded. Upstream explicitly gives no universal termination
guarantee.

Current `KeyboardInterrupt` behavior catches the signal at the top-level reducer
and returns the basis, transformation, and profile accumulated so far. Such a
return:

- is a partial result, not a successful reduction;
- has no weak-LLL, DeepLLL, or BKZ postcondition;
- intends to retain `B0*U=B`, but current updates are not transactional and
  unchecked overflow can invalidate the mathematical relation; and
- does not currently carry an explicit interrupted status in the return type.

There is no current timeout or stagnation result because there is no such
control. A future implementation may add opt-in limits only if it returns a
typed partial status (`interrupted`, `timeout`, `iteration_limit`, or
`stagnation`) and never labels that result reduced. Defaults must remain
unlimited until an algorithm-version decision changes them.

## Output contracts

All contracts below assume exact integer operations, controlled real decisions,
normal termination, and successful independent verification.

| Mode | Structural contract | Reduction contract |
|---|---|---|
| LLL + classical panel | `U` unimodular and `B=B0*U` | global weak-LLL; intended global size reduction, hence BLASter LLL |
| LLL + Seysen panel | same | global weak-LLL plus the specified recursive Seysen-panel transform; no separate global Seysen certificate |
| DeepLLL + classical panel | same | global weak-LLL and intended size reduction; local alternating segments used DeepLLL, but no global DeepLLL certificate |
| DeepLLL + Seysen panel | same | global weak-LLL; no global DeepLLL or Seysen certificate |
| BKZ | same | final embedded DeepLLL-4 segmented pass reaches the corresponding weak-LLL/panel condition; requested heuristic tours completed, but no global BKZ certificate |
| Partial return | relation must be independently checked | no reducedness contract |

The word "intended" matters for the current FastInt64/binary64 implementation:
it lacks the arithmetic evidence needed to certify the mathematical contract.

## Known limitations and ambiguities

- The upstream algorithm is empirical and lacks a general termination or
  correctness proof.
- Current full-rank validation computes exact rank over GF(p) for five primes
  below 2^31. This proves Q-independence exactly and cannot produce the former
  float64 false-negative; it can reject a full-rank basis only when its
  determinant is a nonzero multiple of all five primes (probability negligible
  for non-adversarial input).
- Global termination and local LLL disagree on exact Lovasz equality.
- The two rounding tie rules can produce different transformations at the same
  half-integer.
- Binary64 decisions have no uncertainty interval; QR sign and accumulated error
  may alter comparisons and rounded coefficients.
- The recursive Seysen name is stronger than its checked postcondition.
- DeepLLL's implemented depth predicate differs from the simple prose
  "maximum insertion distance".
- BKZ's Gaussian-heuristic caller/helper disagree on whether their logarithm is
  volume or squared volume.
- Pruning can miss a short vector, and candidates without an easy coefficient
  `+/-1` insertion are discarded.
- The final BKZ result is not certified BKZ-reduced.
- Current signed-int64 operations can overflow without a reliable error.

These are review inputs for later algorithm versions. M1 documents them and does
not change them.

## Traceable sources

- Top-level scheduling and parameter behavior:
  [`src/blaster/blaster.py`](../../src/blaster/blaster.py)
- Predicates and panel transforms:
  [`src/blaster/size_reduction.py`](../../src/blaster/size_reduction.py)
- Segment extraction and commits:
  [`core/blaster.pyx`](../../core/blaster.pyx)
- Local LLL, DeepLLL, SVP, and BKZ:
  [`core/lattice_reduction.cpp`](../../core/lattice_reduction.cpp)
- Enumeration:
  [`core/enumeration.cpp`](../../core/enumeration.cpp) and
  [`core/enumeration.hpp`](../../core/enumeration.hpp)
- Upstream status and examples:
  [official BLASter repository](https://github.com/ludopulles/BLASter)
- Design paper:
  [Towards a modern LLL implementation](https://ir.cwi.nl/pub/35237)
