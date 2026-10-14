# BLASter

BLASter is a research lattice-reduction tool built around a segmented, parallel
LLL-like algorithm. It combines local LLL or DeepLLL, optional BKZ, Seysen or
classical size reduction, and Eigen-backed integer matrix operations.

The project provides:

- an installable Python package;
- a row-oriented NumPy and SageMath API;
- the original column-oriented reduction API;
- an fplll-compatible command-line interface; and
- deterministic correctness tests and smoke benchmarks.

## Status and limitations

BLASter remains a research prototype. It does not guarantee termination or
correct output for every lattice. The default FastInt64 engine uses signed
64-bit integer storage, so both input entries and intermediate transformations
are subject to the int64 range; the exact reference engine
(``engine="reference"`` on ``reduce_basis``) lifts this limit for ordinary LLL
at arbitrary precision. BKZ enumeration is limited to block size 80.

Use the transformation invariant and the reduction checks exposed by the test
suite when evaluating new inputs. Do not treat BLASter as a hardened
cryptographic implementation.

## Upstream provenance

This checkout starts from the official
[ludopulles/BLASter](https://github.com/ludopulles/BLASter) repository and
incorporates upstream `main` through commit
`68225ece2e10d6f50a5a811de63d97583a987dc6`.

The design paper is
[Towards a modern LLL implementation](https://ir.cwi.nl/pub/35237).

Important upstream changes retained by this checkout include:

- pip-installable package layout under `src/blaster/`;
- Cython extension name `blaster._core`;
- a shared `_propagate_update` path for segmented block updates;
- input-reading and stdio fixes;
- removal of the deprecated `Eigen::initParallel()` call;
- the `beta < 40` DeepLLL-4 behavior from upstream PR #6; and
- the weak-LLL predicate without the legacy modulo step.

## Environment

All repository commands in this checkout use the Conda environment `sage106`,
containing SageMath 10.6 and Python 3.11. Verify it with:

```console
make env-check
```

Build requirements are Cython 3 or later, NumPy, cysignals, setuptools, a C++17
compiler, Eigen 3, and OpenMP. On macOS, the compiler must be able to find
`libomp`; Homebrew and conda-forge installations are detected by the current
build configuration. Matplotlib and Pillow are optional and are required only
for animated profiles.

If Eigen is not installed system-wide, install the vendored copy:

```console
make eigen3
```

## Build and install

Build the package extension in place and run the tests:

```console
make cython
make test
```

Build a wheel through the standard `pyproject.toml` workflow:

```console
make wheel
```

Install the current source into `sage106`:

```console
make install
```

The equivalent direct command is:

```console
conda run -n sage106 python -m pip install --no-build-isolation .
```

The upstream `make venv` and `make venv-clean` targets are also retained for
environments that prefer a local virtual environment.

## Python API

`reduce_basis` is the recommended public interface. Basis vectors are rows by
default, matching SageMath and common lattice-tool conventions:

```python
import numpy as np
from blaster import reduce_basis

basis = np.array([
    [10, 0, 0],
    [1, 10, 0],
    [0, 1, 1],
], dtype=np.int64)

reduced, transform, profile = reduce_basis(
    basis,
    return_transform=True,
    lll_size=3,
    cores=1,
    use_seysen=True,
)
assert np.array_equal(transform @ basis, reduced)
```

For a Sage integer matrix, `reduce_basis` returns Sage matrices and preserves
the row-vector transformation convention:

```python
from sage.all import ZZ, matrix
from blaster import reduce_basis

basis = matrix(ZZ, [[10, 0, 0], [1, 10, 0], [0, 1, 1]])
reduced, transform, profile = reduce_basis(basis, return_transform=True, cores=1)
assert transform * basis == reduced
```

The lower-level `reduce` function retains BLASter's original column-vector
convention and always returns NumPy int64 matrices:

```python
from blaster import reduce

transform, reduced, profile = reduce(column_basis, cores=1)
assert np.array_equal(column_basis @ transform, reduced)
```

Set `orientation="columns"` on `reduce_basis` when a type-preserving,
column-oriented interface is needed.

### Exact and Sage-accelerated backends

`reduce_basis` uses the C-accelerated FastInt64 engine by default, which requires
int64 entries and computes Gram-Schmidt in binary64. Pass `engine="reference"`
for an exact arbitrary-precision LLL (ordinary LLL only, any entry magnitude,
only the `delta` parameter; the third return value is a decision transcript
rather than a `TimeProfile`):

```python
from blaster import reduce_basis

big = 1 << 70  # exceeds int64
reduced, transform, transcript = reduce_basis(
    [[big, 0, 0], [0, big, 0], [big // 2, big // 3, 1]],
    engine="reference",
    return_transform=True,
)
```

`verify_reduction` takes a `gso` selector for its exact Gram-Schmidt backend:
`gso="rational"` (the default) uses pure-Python exact arithmetic, while
`gso="sage"` uses Sage's `gram_schmidt` accelerator under the same exact
contract:

```python
from blaster import verify_reduction

rep = verify_reduction(B0, B, U, delta=0.99, gso="sage")
assert rep.verified
```

The exact reference reducer accepts the same selector on its internal
Gram-Schmidt primitive (the algorithm is unchanged and still never calls Sage
LLL/BKZ):

```python
from blaster import lll_reduce_rows

rows, T, transcript = lll_reduce_rows(A0, delta=0.99, gso="sage")
```

Runnable Sage scripts covering LLL, DeepLLL, BKZ, and column-oriented bases are
available in [`examples/sage/`](examples/sage/). Run all of them with:

```console
bash examples/sage/run_all.sh
```

## CLI

The CLI reads and writes fplll matrix syntax. The file format lists basis vectors
as rows; BLASter converts them to its internal column representation.

```console
conda run -n sage106 blaster --input input.lat --output reduced.lat
```

stdin and stdout are used when paths are omitted:

```console
latticegen q 128 64 20 p | conda run --no-capture-output -n sage106 blaster --quiet --profile
```

Common reduction modes are:

```console
conda run -n sage106 blaster -i input.lat -d 4
conda run -n sage106 blaster -i input.lat -b 60 -t 1 -P 2
conda run -n sage106 blaster -i input.lat --no-seysen
```

Use `conda run -n sage106 blaster --help` for the complete parameter list.
Invalid matrices, non-integer or out-of-range entries, incompatible options, and
unsupported BKZ block sizes are reported as CLI errors without a traceback.

The Python I/O layer under the CLI (`blaster.lattice_io`) reads and writes
fplll matrix syntax. By default `parse_lattice`, `read_lattice`, and
`write_lattice` use int64 storage and reject out-of-range entries; pass
`exact=True` to read or write entries of any magnitude as exact Python
integers:

```python
from blaster.lattice_io import parse_lattice, write_lattice

big = 1 << 70  # exceeds int64
basis = parse_lattice(f"[[{big} 1] [2 3]]", qary_order=False, exact=True)
write_lattice(basis, exact=True)  # round-trips without overflow
```

## Tests

```console
make test
```

The suite covers NumPy and Sage inputs, row and column conventions, LLL,
DeepLLL, BKZ, Seysen reduction, parallel segments, deterministic random and
q-ary lattices, transformation invariants, weak LLL checks, int64 overflow,
parameter validation, file I/O, stdin/stdout, and both CLI entry points.

## Benchmarks

Run the fast deterministic baseline from the repository root:

```console
conda run -n sage106 python benchmark/smoke_benchmark.py --dimensions 32 64 --cores 2
conda run -n sage106 python benchmark/smoke_benchmark.py --dimensions 32 64 --cores 2 --seysen
```

The recorded development-machine results are in
[`benchmark/SMOKE_BASELINE.md`](benchmark/SMOKE_BASELINE.md).

The measured Flatter/BLASter behavior and performance comparison is documented
in [`benchmark/FLATTER_VS_BLASTER.md`](benchmark/FLATTER_VS_BLASTER.md). The tools
are treated as different algorithms, not interchangeable runtime backends.

The original comparison scripts require their corresponding external tools and
input directories. Run them from `benchmark/` after installing BLASter:

```console
cd benchmark
conda run -n sage106 python benchmark.py dim 128 lattices lll
```

The scripts use the installed `blaster` command. Set `BLASTER_CMD` to override
it when testing another build.

## Algorithm and arithmetic specifications

The next development stage is specified before core refactoring:

- [`docs/spec/current-behavior-baseline.md`](docs/spec/current-behavior-baseline.md)
  freezes provenance, commands, tests, benchmarks, defaults, and known limits;
- [`docs/spec/blaster-algorithm.md`](docs/spec/blaster-algorithm.md) defines the
  BLASter state machines and pseudocode independently of implementation types;
- [`docs/spec/implementation-map.md`](docs/spec/implementation-map.md) maps each
  reduction step to Python, Cython, C++, tests, and risks; and
- [`docs/spec/arithmetic-model.md`](docs/spec/arithmetic-model.md) defines
  FastInt64, CheckedInt64, Multiprecision, escalation, and verification.

The accepted architecture decisions are indexed in
[`docs/adr/`](docs/adr/). Flatter and fpylll/fplll remain benchmark,
interoperability, and research-comparison subjects. They are not overflow or
precision fallbacks for BLASter.

## Debug build

Build the Cython/C++ extension with address and undefined-behavior sanitizers:

```console
make cython-gdb
```

Sanitizer runtime setup is compiler- and platform-specific. The debug build does
not change the Python API or CLI entry points.
