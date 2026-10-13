# Current behavior baseline (M0)

Status: current on 2026-08-13 after synchronizing upstream `main` through
`68225ec` and reapplying the local API, validation, test, and documentation
layer. The Git `HEAD` reference still points at `27ad3a4`; the synchronized
content is present in the working tree and index as an uncommitted merge result.

## Provenance and scope

- Project lineage: [ludopulles/BLASter](https://github.com/ludopulles/BLASter).
- Configured fork: `git@github.com:GaloisHLee/BLASter.git`.
- Clean Git baseline in this checkout: `27ad3a4ea6e574b10bb3e5eaf84049c1f83a49c6`
  (`Fix stdin bug & allow output to stdout`).
- Official upstream `HEAD` observed with `git ls-remote` on 2026-08-08:
  `68225ece2e10d6f50a5a811de63d97583a987dc6`.
- Upstream content merged into the worktree on 2026-08-13 without creating a
  commit. The merge target is `upstream/main` at
  `68225ece2e10d6f50a5a811de63d97583a987dc6`.
- The current worktree, including its uncommitted packaging, API, examples,
  benchmark, and test changes, is the authority for this baseline.

The upstream README calls BLASter a proof of concept, explicitly declines to
guarantee termination or correctness on every lattice, does not support large
entries, and invites a robust new implementation based on its ideas. This
baseline retains those limitations rather than silently upgrading the project
status.

The worktree differs from the clean Git baseline mainly by:

- moving the Python modules into the installable `src/blaster/` package;
- adding strict input and parameter validation plus row-oriented NumPy/Sage API;
- adding standard package metadata and a `blaster` console entry point;
- adding deterministic tests and executable Sage examples; and
- adding smoke and external-algorithm comparison benchmarks.

The local layer now sits on top of upstream's pip package layout and core
changes. Upstream changes include `blaster._core`, a shared
`_propagate_update` routine, input-reading fixes, removal of the deprecated
`Eigen::initParallel()` call, the `beta < 40` DeepLLL-4 path, and the weak-LLL
predicate without its legacy modulo step. Local validation can reject inputs
that the original script accepted without checking.

Reproduce the provenance and worktree comparison without changing files:

```console
git log -1 --format='%H%n%ad%n%s' --date=iso-strict
git status --short
git diff --stat
git diff -- core
git diff HEAD upstream/main --stat
git ls-remote https://github.com/ludopulles/BLASter.git HEAD
```

## Frozen environment

All repository commands use the Conda environment `sage106`.

| Component | Observed value |
|---|---|
| SageMath | 10.6 (release 2025-03-31) |
| Python | 3.11.13 |
| NumPy | 2.3.0 |
| Cython | 3.1.4 |
| Platform | macOS 15.7.9, Apple arm64 |
| Package version | BLASter 1.0 |

Reproduce the environment report:

```console
make env-check
conda run -n sage106 python -c \
  'import platform, sys, numpy, Cython; print(platform.platform()); print(sys.version.split()[0]); print(numpy.__version__); print(Cython.__version__)'
```

## Build and test baseline

The authoritative build and full-test command is:

```console
make test
```

On 2026-08-08, before the upstream sync, the checkout rebuilt
`blaster.blaster_core` and completed all 24 tests in 14.246 seconds with `OK`.
After the upstream sync the authoritative command rebuilds `blaster._core`; the
post-sync verification result is:

```text
Ran 24 tests in 91.698s
OK
```

That run includes all NumPy/Sage API, I/O, CLI, representative-lattice, and six
Sage example tests. The Sage examples require write access to `~/.sage/cache`,
so a sandboxed run can fail with `PermissionError` even though the code is
correct.

- row- and column-oriented NumPy bases and a rectangular basis;
- Sage `ZZ` matrix input/output;
- LLL, DeepLLL, segmented Seysen reduction, and a small BKZ run;
- deterministic random and q-ary lattices;
- transformation, determinant, and weak-LLL checks;
- input/parameter rejection and the signed-int64 input boundary;
- fplll-format files, stdin/stdout, and both CLI entry points; and
- all six scripts in `examples/sage/`, including the external comparison and
  large-integer boundary scripts when Flatter is installed.

Run the Sage scripts directly with:

```console
bash examples/sage/run_all.sh
```

The baseline proves only the assertions present in those tests. In particular,
the determinant assertions use floating-point `numpy.linalg.det` in the NumPy
tests, intermediate int64 overflow is not detected, and no independent global
DeepLLL or BKZ certificate is checked.

## Public behavior

The recommended Python/Sage call is `reduce_basis`. It treats vectors as rows by
default and returns `(B_red, T, profile)` satisfying `T * B0 == B_red` when
`return_transform=True`. `orientation="columns"` instead returns
`(B_red, U, profile)` with the internal relation `B0 * U == B_red`. Sage integer
input produces Sage integer matrices.

The lower-level `reduce` call always uses column vectors and NumPy int64 arrays.
Its API defaults are:

| Parameter | Default | Meaning |
|---|---:|---|
| `lll_size` | 64 | local LLL/DeepLLL segment width |
| `delta` | 0.99 | Lovasz parameter, with `0.25 < delta < 1` |
| `cores` | 1 | OpenMP/Eigen thread count |
| `depth` | 0 | ordinary local LLL; positive selects local DeepLLL |
| `use_seysen` | false | classical divide-and-conquer size reduction |
| `beta` | none | no BKZ |
| `bkz_tours` | 8 | target tours when BKZ is enabled |
| `bkz_size` | max(`lll_size`, `beta`) | segmented BKZ width, capped at rank |
| `bkz_prog` | `beta` | progressive-beta increment |

When `beta` is set and smaller than 40, upstream semantics run DeepLLL-4 rather
than starting a pruned BKZ tour. The public `depth` and `beta` options remain
mutually exclusive.

The CLI has the same numerical defaults except that Seysen reduction is enabled
unless `--no-seysen` is passed, and its requested core count defaults to half of
the logical CPUs before being capped for the basis and segment size.

Reproduce the surface:

```console
PYTHONPATH=src conda run -n sage106 python -c 'import numpy as np; from blaster import reduce_basis; B=np.array([[10,0,0],[1,10,0],[0,1,1]],dtype=np.int64); R,T,_=reduce_basis(B,return_transform=True,cores=1); assert np.array_equal(T@B,R)'
conda run -n sage106 sage examples/sage/01_basic_lll.sage
conda run -n sage106 blaster --help
conda run -n sage106 blaster --input input.lat --output reduced.lat
```

The CLI performs a final `basis @ transform == reduced` comparison in int64. A
passing comparison is not an overflow proof because all three operands may
already contain results of unchecked signed-integer operations.

## Deterministic smoke baseline

Commands:

```console
conda run -n sage106 python benchmark/smoke_benchmark.py --dimensions 32 64 --cores 2 --lll-size 32
conda run -n sage106 python benchmark/smoke_benchmark.py --dimensions 32 64 --cores 2 --lll-size 32 --seysen
```

The 2026-08-08 re-run produced the following quality and iteration values. Wall
time is machine noise and is recorded only as a smoke indicator.

| Kind | n | Seysen | Iterations | RHF | Wall time |
|---|---:|---:|---:|---:|---:|
| random | 32 | no | 2 | 1.000000 | 0.001426 s |
| q-ary | 32 | no | 2 | 1.014253 | 0.001885 s |
| random | 64 | no | 5 | 1.000000 | 0.005922 s |
| q-ary | 64 | no | 12 | 1.020563 | 0.016486 s |
| random | 32 | yes | 2 | 1.000000 | 0.001056 s |
| q-ary | 32 | yes | 2 | 1.014253 | 0.001602 s |
| random | 64 | yes | 5 | 1.000000 | 0.004027 s |
| q-ary | 64 | yes | 12 | 1.020563 | 0.011146 s |

The earlier recorded numbers and exact generator/check definitions remain in
[`benchmark/SMOKE_BASELINE.md`](../../benchmark/SMOKE_BASELINE.md) and
[`benchmark/smoke_benchmark.py`](../../benchmark/smoke_benchmark.py).

The 2026-08-13 post-sync run of the same commands, using the local source
through `PYTHONPATH=src`, produced identical iteration and RHF values with
different wall times:

| Kind | n | Seysen | Iterations | RHF | Wall time |
|---|---:|---:|---:|---:|---:|
| random | 32 | no | 2 | 1.000000 | 0.002259 s |
| q-ary | 32 | no | 2 | 1.014253 | 0.002864 s |
| random | 64 | no | 5 | 1.000000 | 0.010166 s |
| q-ary | 64 | no | 12 | 1.020563 | 0.028813 s |
| random | 32 | yes | 2 | 1.000000 | 0.001615 s |
| q-ary | 32 | yes | 2 | 1.014253 | 0.002777 s |
| random | 64 | yes | 5 | 1.000000 | 0.007232 s |
| q-ary | 64 | yes | 12 | 1.020563 | 0.019560 s |

The upstream weak-LLL change did not alter the iteration counts or printed RHF
values on this smoke corpus.

## External-algorithm observations

Flatter and fpylll/fplll are comparison subjects, not arithmetic backends,
fallbacks, or specification oracles for BLASter. Their algorithms, stopping
criteria, precision policies, and output interfaces differ from BLASter's.

The retained Flatter comparison uses deterministic 20-bit q-ary inputs, two
threads, and three repetitions. Its recorded medians are:

| n | BLASter time / RHF | Flatter time / RHF |
|---:|---:|---:|
| 64 | 0.106 s / 1.015968 | 0.433 s / 1.018915 |
| 128 | 0.197 s / 1.019934 | 2.947 s / 1.019749 |
| 256 | 0.667 s / 1.021682 | 23.197 s / 1.019411 |
| 512 | 4.564 s / 1.013630 | 101.295 s / 1.013630 |

The 129-bit, dimension-16 boundary example records Flatter completing in about
0.09 seconds while BLASter rejects the input before reduction because it does
not fit signed int64. These are observations about two distinct algorithms on a
narrow workload, not a routing rule. Reproduction details and raw samples are in
[`benchmark/FLATTER_VS_BLASTER.md`](../../benchmark/FLATTER_VS_BLASTER.md) and
[`benchmark/flatter_vs_blaster.csv`](../../benchmark/flatter_vs_blaster.csv).

## Frozen limitations

- Input entries are range-checked, but transformations and integer matrix
  products use unchecked C++ `long long`; overflow can be silent or undefined.
- QR, reduction predicates, panel solves, profile metrics, and enumeration
  bounds use binary64 without an error bound or precision escalation.
- Core reduction rounds half-integers away from zero; Python panel reduction
  rounds ties to even. This site-specific behavior is part of the current
  baseline.
- The weak-LLL checker now follows upstream `68225ec` and compares
  `R[i,i+1]^2 + R[i+1,i+1]^2` directly, without reducing the off-diagonal
  coefficient modulo `R[i,i]`.
- Full-rank validation computes exact rank over GF(p) for five primes below
  2^31 (modular Gaussian elimination). A full-rank reading modulo any one prime
  proves Q-independence, so a full-rank basis is never misread as deficient; a
  basis whose determinant is a nonzero multiple of all five primes is still
  rejected. No floating-point rank test is used.
- BKZ enumeration is capped at dimension 80 and uses heuristic pruning.
- Normal LLL mode has no timeout, iteration limit, or stagnation detector.
- `KeyboardInterrupt` returns a partial result without a reducedness guarantee.
- Global termination checks weak LLL only. DeepLLL and BKZ modes do not return an
  independently certified global DeepLLL- or BKZ-reduced basis.
- Upstream itself gives no universal termination or correctness guarantee.

No later implementation may treat this baseline as permission to preserve a
known arithmetic error silently. It is the comparison point from which checked
and multiprecision behavior must report deliberate deviations.
