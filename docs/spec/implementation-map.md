# BLASter specification-to-implementation map (M1)

Version: 0.2 review candidate, mapped against the 2026-08-13 upstream-synced
worktree.

This map connects each step in
[`blaster-algorithm.md`](blaster-algorithm.md) to executable code and current
tests. A dash means that the layer delegates the step without implementing it.
Test presence records exercised behavior, not a proof of the mathematical
postcondition.

Layer paths used by the tables are:

- Python reducer/API: [`src/blaster/`](../../src/blaster/), especially
  `blaster.py`, `size_reduction.py`, `api.py`, `_matrix.py`, and `app.py`;
- Cython bridge: [`core/blaster.pyx`](../../core/blaster.pyx);
- C++ reducers and kernels: [`core/lattice_reduction.cpp`](../../core/lattice_reduction.cpp),
  [`core/enumeration.cpp`](../../core/enumeration.cpp),
  [`core/enumeration.hpp`](../../core/enumeration.hpp),
  [`core/pruning_params.cpp`](../../core/pruning_params.cpp), and
  [`core/eigen_matmul.cpp`](../../core/eigen_matmul.cpp); and
- tests: [`tests/test_api.py`](../../tests/test_api.py),
  [`tests/test_reduction_cases.py`](../../tests/test_reduction_cases.py),
  [`tests/test_io_cli.py`](../../tests/test_io_cli.py), and
  [`tests/test_sage_examples.py`](../../tests/test_sage_examples.py).

## Naming and orientation

| Specification concept | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Canonical column basis `B0`, `B`, right transform `U` | `blaster.reduce` | typed `B_red`, `U` arguments | `ZZ *U` local transform | `test_reduce_columns_preserves_lattice_and_input` | int64 relation can agree after common corruption |
| Row API `T*A0=A` with `T=U^T` | `api.reduce_basis` | - | - | `test_reduce_basis_defaults_to_row_vectors`, Sage and rectangular tests | transpose convention can be lost by new entry points |
| Integer input conversion | `_matrix.as_int64_matrix` | `NP_ZZ=np.int64` | `ZZ=long long` in `types.hpp` | rejection tests and large-integer Sage example | only input is checked; no intermediate range proof |
| Full-column-rank precondition | `blaster._validated_parameters` | - | reducers assume nonzero diagonal | invalid-shape and wide-dynamic-range rank tests | exact GF(p) rank; a full-rank reading proves Q-independence, a determinant divisible by all five primes is still rejected |
| Initialize `B=B0`, `U=I`, panel scratch `W=I` | `blaster.reduce` | - | local `init_U` for each segment | rank-one and transformation tests | panel scratch is reused and relies on full overwrite of its upper triangle |

## Segmented LLL state machine

| Specification step | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Select LLL or DeepLLL local mode | `lll_reduce`, `red_fn` | `block_lll`, `block_deep_lll` | `lll_reduce`, `deeplll_reduce` | basic LLL and `test_deep_lll_preserves_lattice` | final mode guarantee is only global weak LLL |
| QR checkpoint A before local segments | `lll_reduce`: first `np.linalg.qr(..., mode='r')` | - | - | all reduction cases indirectly | binary64 has no error estimate |
| Toggle first half-offset then zero | `lll_reduce`, `offset` assignment | receives `offset` | - | q-ary parallel-segment test indirectly | exact offset sequence has no direct unit test |
| Construct disjoint segments | - | `num_blocks`, `i`, `w` in each `block_*` | receives one width `N` | q-ary parallel-segment test | prefix before nonzero offset is intentionally untouched |
| Copy diagonal `R[S,S]` into private local storage | - | `R_sub` allocation and `memcpy` loops | consumes flat `R` block | debug/profile paths exercise copies | layout depends on consistent row-major indexing |
| Compute segments concurrently | - | OpenMP `prange` in each `block_*` | one local reducer per segment | two-core q-ary test | thread count is process-global; no deterministic transcript test |
| Commit each local `V_S` to `U[:,S]` and `B[:,S]` | - | `_propagate_update`; increasing `block_id`; `ZZ_right_matmul_strided` | `eigen_right_matmul` | all transformation tests | multiplication is unchecked signed `long long` |
| Optional local LLL checker | `lll_reduce` under `debug` | debug-only copy of reduced `R_sub` back to `R` | - | CLI `--profile` tests | strict global equality rule can disagree with local LLL acceptance |
| QR checkpoint B after local commits | `lll_reduce`: second `np.linalg.qr` | - | - | all reduction cases indirectly | recomputation is approximate but not precision-controlled |
| Select classical or Seysen panel | `lll_reduce` | `FT_matmul`, integer panel multiplication helpers | Eigen integer kernels | classical, Seysen, and DeepLLL tests | API defaults classical while CLI defaults Seysen |
| Commit panel transform to `U` and `B` | `ZZ_right_matmul(U,W)` and `ZZ_right_matmul(B,W)` | `ZZ_right_matmul` | `eigen_right_matmul` | transformation tests | two separate in-place products are not one transaction |
| Evaluate global stop condition | `is_weakly_lll_reduced(R,delta)` | - | - | every representative reduction case | no interval comparison, timeout, or stagnation handling |
| Record iteration/profile | `TimeProfile.tick`, `stats.get_profile`, tracers | - | - | iteration-count and CLI profile tests | metrics are approximate observations, not certificates |

## Local LLL primitives

| Specification step | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Initialize local identity `V` | - | supplies allocated `U_sub` | `init_U` | all LLL cases indirectly | none under exact arithmetic |
| Size-reduce column `k` against prior columns | - | - | `_lll_reduce`, `size_reduce` | small deterministic reductions | division and `llround` use binary64; integer coefficient may exceed `long long` |
| Apply integer column addition | - | - | `alter_basis` | transformation equality indirectly | `U += z*U` is unchecked; `R` update is approximate |
| Test local Lovasz condition | - | - | `_lll_reduce` | weak-LLL assertions indirectly | non-strict local comparison differs from the strict global checker |
| Adjacent swap and Givens restoration | - | - | `swap_basis_vectors` | random lattices exercise swaps | zero/underflowed rotation norm and accumulated real error are unchecked |
| Move local index backward/forward | - | - | `_lll_reduce` | random lattices indirectly | local termination is inherited from floating decisions, not certified |

## Panel reduction and predicates

| Specification step | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Build balanced reduction tree | `size_reduction.__reduction_ranges` | - | - | size/Seysen integration tests | tree order affects binary64 rounding; no direct scheduler test |
| Build deferred Babai propagation ranges | `size_reduction.__babai_ranges` | - | - | classical integration tests | algebraic equivalence assumes exact arithmetic |
| Babai coefficient rows, descending | `nearest_plane`, `np.rint` | `FT_matmul` | - | classical integration tests | ties-to-even differs from core rounding; float-to-int cast can overflow |
| Classical two-column leaf | `size_reduce`, built-in `round` | - | - | basic LLL tests | host-language ties-to-even is implicit unless wrapped by arithmetic layer |
| Classical cross-panel update | `size_reduce`, `nearest_plane` | `FT_matmul`, `ZZ_left_matmul_strided` | `eigen_left_matmul` | random and basic tests | real panel and exact transform can diverge after rounding error/overflow |
| Seysen two-column leaf | `seysen_reduce`, built-in `round` | - | - | Seysen q-ary/DeepLLL tests | same tie and division risks |
| Seysen rounded inverse cross panel | `seysen_reduce`, `np.linalg.inv`, `np.rint` | `FT_matmul`, `ZZ_left_matmul_strided` | `eigen_left_matmul` | Seysen q-ary/DeepLLL tests | explicit inverse is unstable; no residual or condition check |
| Weak-LLL predicate | `is_weakly_lll_reduced` | - | - | all representative cases | upstream predicate compares `v^2+w^2` directly; strict boundary has no uncertainty handling |
| Size-reduced predicate | `is_size_reduced` | - | - | debug local checks | not checked on normal return |
| Combined LLL predicate | `is_lll_reduced` | - | - | debug local checks | checker is not an independent verifier |

## DeepLLL

| Specification step | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Public depth selection | `_validated_parameters`, `lll_reduce` | `block_deep_lll(depth,...)` | `deeplll_reduce` | `test_deep_lll_preserves_lattice`, Sage example 02 | no direct depth-boundary trajectory test |
| LLL preprocessing on prefix | - | - | `_deeplll_reduce` calls `_lll_reduce` | DeepLLL integration tests | shares all local LLL arithmetic risks |
| Compute projected norm | - | - | `_deeplll_reduce`, `proj_norm_sq` | DeepLLL integration tests | sum of squares can overflow/underflow in binary64 |
| Apply implemented depth predicate | - | - | `(i < depth || i >= k-depth)` | no focused test | differs from simple maximum-distance wording |
| Deep insertion by adjacent swaps | - | - | `_deeplll_reduce` swap loop | DeepLLL integration tests | only local segments are deep-reduced |
| BKZ-internal fixed depth four | `reduce` passes `depth=4` | `block_bkz` | `bkz_reduce` and `svp` | small BKZ test and Sage example 03 | public `depth` is prohibited with `beta`, so this policy is implicit |

## Segmented BKZ and SVP

| Specification step | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Validate beta/tours/segment/progression | `_validated_parameters` | fixed storage assumes beta <= 80 | `MAX_ENUM_N=80` | invalid-parameter tests | upper bound is a compile-time implementation limit |
| Generate progressive beta sequence | `_progressive_betas` | - | - | no focused unit test | values below 40 return an empty BKZ schedule and select DeepLLL-4 |
| Initial DeepLLL-4 segmented reduction | `bkz_reduce` calls `lll_reduce` | `block_deep_lll` | `deeplll_reduce` | small BKZ test | may not terminate before finite tours begin |
| Derive partition offset from front | `bkz_reduce`, `cur_front % beta` | receives offset | - | BKZ integration test | `front` is not one selected SVP window |
| Run local BKZ per segment | `bkz_reduce` | `block_bkz` | `bkz_reduce` | small BKZ test | short tail may take beta>N HKZ-style branch |
| Local DeepLLL-4 preprocessing | - | - | `bkz_reduce` | small BKZ test | no global deep certificate |
| Select fixed-beta or suffix windows | - | - | `bkz_reduce` branch on `beta > N` | no focused test | naming the suffix branch HKZ overstates certification |
| Compute log-volume argument | - | - | `svp`, sum of `log(RSQ)` | no focused test | caller supplies log squared volume to helper documented for log volume |
| Compute GH-derived radius | - | - | `gh_squared`, `safe_gh_squared`, `svp` | no focused test | heuristic bound and binary64 transcendental functions |
| Select pruning vector | - | - | `get_pruning_coefficients` in `pruning_params.cpp` | no focused test | hard-coded heuristic success probability |
| Build GSO inputs for enumeration | - | - | `enumeration` | small BKZ indirectly | divisions and squared norms are unbounded-error binary64 |
| Traverse pruned enumeration tree | - | - | `lattice_enum_t::enumerate_recursive` | small BKZ indirectly | core rounding, equality, and coefficient casts are unchecked |
| Accept best found candidate | - | - | `enumeration` return and `svp` norm test | no focused test | pruning can return no candidate despite a shorter vector |
| Require easy `+/-1` insertion | - | - | `svp`, last-nonzero checks | no focused test | useful candidates with other final coefficient are discarded |
| Insert candidate and move to front | - | - | `svp`, `alter_basis`, swap loop | small BKZ indirectly | exact transform can overflow; real state is approximate |
| DeepLLL-4 prefix after every oracle call | - | - | final call in `svp` | small BKZ indirectly | happens even when no insertion occurred |
| Panel reduction after segmented BKZ | `bkz_reduce` | same panel kernels | Eigen integer kernels | small BKZ test | same panel risks as LLL mode |
| Advance front and tour | `bkz_reduce` | - | - | no focused unit test | strict `front+beta>n` is easy to misread as `>=` |
| Segmented LLL after every BKZ iteration | unconditional call at end of loop body | `block_deep_lll` | `deeplll_reduce` | small BKZ indirectly | source comment says final/end, executable behavior is per iteration |

## Completion and public surfaces

| Specification step | Python | Cython | C++ | Current tests | Known risk |
|---|---|---|---|---|---|
| Catch user interruption and return partial state | `reduce`, nested `KeyboardInterrupt` handler | `cysignals` brackets local calls | - | no interruption test | result type does not distinguish partial from normal completion |
| Close logfile on all exits | `reduce`, `finally` | - | - | no focused test | animation errors occur after reduction and are a separate surface failure |
| Return transformation, reduced basis, profile | `reduce`; adapted by `reduce_basis` | - | - | API tests | return contract overstates LLL in the current docstring for non-classical modes |
| Read/write fplll row syntax | `lattice_io`, CLI `_run` | - | - | I/O and CLI tests | format conversion transposes at boundary |
| CLI post-run transformation comparison | `app._run` | `ZZ` results | Eigen results | CLI tests | comparison is not independent and uses already-computed int64 state |

## Arithmetic operation ownership

The following ownership summary prevents an implementation refactor from moving
an operation across a precision boundary accidentally.

| Operation | Current owner | Required category |
|---|---|---|
| Store `B`, `U`, local and panel transforms | NumPy int64 / C++ `long long` | exact integer |
| Compose/commit transforms | Eigen kernels exposed through Cython | exact integer |
| QR and Givens rotation | NumPy LAPACK / C++ binary64 | controlled real |
| Local predicates and projected norms | C++ binary64 | controlled real decision |
| Babai/Seysen coefficients | NumPy binary64 plus integer rounding | controlled real decision producing exact integer |
| Enumeration centers, bounds, and pruning | C++ binary64 | controlled real decision producing exact integer coefficients |
| Profile/RHF/slope/potential | Python binary64 | approximate diagnostic |
| Transformation and lattice verification | current NumPy/Sage tests | independent exact verifier in the target design |

The complete interface and recovery contract is in
[`arithmetic-model.md`](arithmetic-model.md).

## Coverage gaps that gate a reference implementation

Before a Sage reference reducer is accepted, review must add evidence for these
currently unmapped guarantees:

- exact rank, determinant, unimodularity, and `B0*U=B` checks independent of the
  reducer's machine-integer kernels;
- half-integer tests for both specified tie rules;
- exact Lovasz equality and near-boundary precision-escalation tests;
- transactional checked-int64 overflow tests for column addition and every
  matrix-product shape;
- direct offset, reduction-tree, front/tour, and progressive-beta schedule tests;
- explicit normal-versus-partial completion status tests;
- a definition-level classical size-reduction postcondition test;
- transcript tests for local DeepLLL depth selection and BKZ easy insertion; and
- tests showing that a verifier can reject a deliberately corrupted reducer
  result.

These gaps do not invalidate the M0 behavior baseline; they delimit what the
current suite actually establishes.
