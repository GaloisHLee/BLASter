# Multi-core switch: `cores="auto"` for the fast engine

Status: accepted design (M-2). Version 0.1. Date 2026-10-15.

This document defines the explicit multi-core switch on the FastInt64 engine.
It gives `cores` two modes of use — automatic detection of the maximum core
count, or an explicit caller-supplied count — and makes automatic the default,
across both the library `reduce()` API and the `blaster` command line. It is
the spec that M-2 is implemented against: read it before changing how `cores`
is resolved.

## 1. What this is

The FastInt64 engine (`src/blaster/blaster.py` → `blaster._core`) is segmented:
`block_lll` / `block_deep_lll` / `block_bkz` reduce basis segments concurrently
under a Cython `prange`, and the Eigen integer matmuls use
`Eigen::setNbThreads`. Both are driven by one OpenMP thread count, set once via
`set_num_cores(cores)` → `omp_set_num_threads(cores)` + `eigen_init(cores)`
(`core/blaster.pyx:29-31`).

Before M-2 that count was only ever an explicit integer, and the default was
`cores=1` (single-threaded). This spec replaces that single mode with two:

1. **auto** — detect and use the maximum core count (`os.cpu_count()`, all
   logical CPUs, no capping).
2. **explicit** — the caller supplies `cores=N` with `N >= 1`.

Automatic is the default. `cores` remains a fast-engine-only parameter: the
reference engine (exact, arbitrary-precision) is single-threaded and is
unchanged.

## 2. Exact contract — library `reduce()`

`reduce()` (`src/blaster/blaster.py:288`) changes its signature from
`cores: int = 1` to `cores: Union[int, str] = "auto"`. Resolution happens
inside `_validated_parameters` via a new helper, so the value that reaches
`set_num_cores` is always a concrete `int >= 1`.

### 2.1 `_resolve_cores(cores) -> int`

```
if cores is a str:
    if cores.strip().lower() == "auto":
        detected = os.cpu_count()
        return detected if (detected is not None and detected >= 1) else 1
    raise ValueError("cores must be 'auto' or an integer")
otherwise:
    return _integer_parameter("cores", cores, 1)
```

- `"auto"` is matched case-insensitively and whitespace-trimmed: `"auto"`,
  `"Auto"`, `"AUTO"`, `" auto "` all resolve to the same count.
- `os.cpu_count()` returns `None` on platforms where it is unavailable; the
  helper falls back to `1` in that case. Otherwise the count is used as-is
  (all logical cores — "最大核数", no `//2` and no block-count cap).
- Any integer path reuses `_integer_parameter("cores", cores, 1)`, preserving
  the existing rules: `bool` and non-`Integral` values raise
  `ValueError("cores must be an integer")`, and `N < 1` raises
  `ValueError("cores must be at least 1")`.
- Any other string (e.g. `"bogus"`) raises `ValueError` whose message contains
  `"cores"`.

### 2.2 Wiring

- `src/blaster/blaster.py:240`: replace
  `cores = _integer_parameter("cores", cores, 1)` with
  `cores = _resolve_cores(cores)`.
- `_validated_parameters` therefore returns a resolved `int` in its `cores`
  slot, and `reduce()`'s `set_num_cores(cores)` at line 318 is unchanged.
- `reduce_basis()` (`src/blaster/api.py`) forwards `**parameters` to `reduce()`
  for the fast engine, so it gains `"auto"` automatically; no signature change
  there.

## 3. Exact contract — CLI `blaster`

`src/blaster/app.py` changes in three places:

1. `--cores` / `-j` accepts either `"auto"` (case-insensitive) or an integer.
   Implement with a small `argparse` `type` callable that returns `"auto"` for
   the string and `int(value)` otherwise. Default becomes `"auto"`.
2. The old default `max(1, cpu_count() // 2)` and the clamp in `_run`
   (`min(args.cores, ceil(rank / max(2, lll_size)), cpu_count() // 2)`,
   `src/blaster/app.py:113-115`) are removed. `"auto"` means all cores, and an
   explicit `--cores N` means `N`, not `min(N, half)`.
3. `_run` passes `args.cores` straight through to
   `reduce(..., cores=args.cores)`, which resolves it per §2.1.

The CLI's `multiprocessing.cpu_count` import may be dropped if it becomes
unused.

## 4. Semantics and invariants

- **Parallelism does not change the output.** The reduction result is a
  function of the lattice and the reduction parameters, not of the OpenMP
  thread count. For any basis, `reduce(B, cores="auto")`, `reduce(B, cores=1)`
  and `reduce(B, cores=N)` return the same reduced basis and transform. This is
  the determinism invariant the acceptance tests pin.
- **`omp_set_num_threads` is process-global.** The last `reduce()` call in a
  process sets the OpenMP thread count for that process. This is unchanged from
  before M-2 and remains documented behavior.
- **Reference engine unaffected.** `engine="reference"` never reads `cores`;
  passing `cores` (or `cores="auto"`) with `engine="reference"` is still
  rejected by `_reduce_basis_reference`'s parameter allow-list, exactly as
  `cores=1` was before.

## 5. Worked examples

Two-column basis `B = [[10, 0], [0, 1]]` (columns), `lll_size=2`,
`delta=0.99`.

- `reduce(B, cores=1)` → single-threaded, reduced basis `B · U`.
- `reduce(B, cores=4)` → 4 OpenMP threads, **same** `B · U`.
- `reduce(B, cores="auto")` → `os.cpu_count()` threads, **same** `B · U`.
- `reduce(B)` → default `"auto"`, **same** `B · U`.
- `reduce(B, cores=0)` → `ValueError: cores must be at least 1`.
- `reduce(B, cores="bogus")` → `ValueError` mentioning `cores`.

CLI (fplll input on stdin):

- `blaster --no-seysen --cores 1` → single-threaded, exit 0.
- `blaster --no-seysen --cores auto` → auto, exit 0.
- `blaster --no-seysen` → default auto, exit 0.
- `blaster --no-seysen --cores bogus` → non-zero exit, error mentions `cores`.

## 6. Acceptance driver (drives the tests)

Tests live in `tests/test_api.py` (unit + end-to-end) and
`tests/test_io_cli.py` (CLI). Each must be written first and observed to fail
before the implementation exists (TDD).

1. `_resolve_cores("auto") == os.cpu_count()` when `os.cpu_count() >= 1`;
   also `_resolve_cores("Auto")`, `_resolve_cores("AUTO")`,
   `_resolve_cores(" auto ")` equal the same value.
2. `_resolve_cores(4) == 4`, `_resolve_cores(1) == 1`.
3. `_resolve_cores(0)` raises `ValueError` matching `"cores"`;
   `_resolve_cores(True)` raises `ValueError` matching `"cores"`;
   `_resolve_cores("bogus")` raises `ValueError` matching `"cores"`.
4. Determinism: `reduce(B, cores="auto")` returns the same basis and transform
   as `reduce(B, cores=1)` (and as `reduce(B, cores=2)`), elementwise.
5. Default-is-auto: `reduce(B)` (no `cores`) returns the same basis as
   `reduce(B, cores=os.cpu_count())`; this pins the default, not a specific
   literal.
6. Existing invalid-parameter table (`tests/test_api.py:140`,
   `{"cores": 0}`) keeps passing — the explicit-integer error path is
   preserved.
7. CLI: `--cores auto` and omitted `--cores` both exit 0 and reduce;
   `--cores N` exits 0; `--cores bogus` exits non-zero with `cores` in stderr.

## 7. Out of scope

- No change to the FastInt64 arithmetic, the `prange` structure, or the Eigen
  matmuls; only the *count* of threads and its resolution change.
- No auto-capping of `cores` to the number of segments or to half the cores —
  "auto" is explicitly the maximum.
- No per-call `os.cpu_count()` re-detection on every `reduce()` invocation:
  `"auto"` re-reads `os.cpu_count()` at each call, which is the intended
  semantics (a caller can also pin a number).
- No change to the reference engine or to `engine="reference"` parameter
  validation.
