# Multi-core switch (`cores="auto"`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit multi-core switch to the FastInt64 engine: `cores="auto"` (default, max cores) or `cores=N` (explicit), in both the library and the CLI.

**Architecture:** Resolution happens once, in a new `_resolve_cores(cores)` helper inside `_validated_parameters`, so `set_num_cores` still receives a concrete `int` and `core/blaster.pyx` + Eigen are untouched. The CLI converts `--cores` to either `"auto"` or an `int` and passes it straight through to `reduce()`.

**Tech Stack:** Python 3 (`unittest`), Cython/OpenMP `_core` (unchanged), `os.cpu_count()` for detection.

**Spec:** `docs/spec/multicore-switch.md` (binding authority; the plan argues from it).

## Global Constraints

- Default `cores` = `"auto"` = `max(1, os.cpu_count())`; explicit `cores=N` requires `N >= 1`. (`os.cpu_count() -> None` falls back to `1`.)
- Parallelism must not change the reduced result: `cores=1` / `cores=N` / `cores="auto"` return identical bases/transforms.
- Reference engine (`engine="reference"`) is single-threaded and rejects `cores`; only the fast engine resolves it.
- Tests use `unittest`, not pytest.
- TDD: every production change is preceded by a failing test that is watched to fail.
- Never `git add t.py`.
- Commit messages end with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

The five input classes most likely to bite a user, not covered by the happy-path tests:

1. `os.cpu_count()` returns `None` (rare platform) → `_resolve_cores("auto")` must fall back to `1`, not crash. → Task 1 test `test_resolve_cores_falls_back_to_one_when_cpu_count_is_none`.
2. A numeric string `"4"` passed to the library → must raise (only `"auto"` is special; the CLI's own type callable handles digit strings, but the library must reject them). → Task 1 test `test_resolve_cores_rejects_invalid_values` includes `"4"`.
3. `cores=True` (bool) → must raise, not coerce to `1`. → Task 1 test `test_resolve_cores_rejects_invalid_values` includes `True`.
4. `--cores 0` on the CLI (explicit int below minimum) → must exit non-zero with `cores` in stderr, not silently clamp to `1`. → Task 2 test `test_cli_rejects_cores_below_minimum`.
5. The row-oriented `reduce_basis` must forward `cores="auto"` (the spec only exercises `reduce` directly). → Task 1 test `test_reduce_basis_forwards_cores_auto`.

---

### Task 1: Library `_resolve_cores` + `reduce()` default

**Files:**
- Modify: `src/blaster/blaster.py` (imports; add `_resolve_cores` after `_integer_parameter` ~line 180; line 240; signature line 289)
- Test: `tests/test_api.py` (add `MultiCoreSwitchTests` class)

**Interfaces:**
- Consumes: `_integer_parameter` (existing, line 174), `set_num_cores` (existing import from `._core`).
- Produces: `_resolve_cores(cores) -> int` (library-internal, imported directly by tests); `reduce(..., cores: Union[int, str] = "auto")`.

- [ ] **Step 1: Write the failing tests**

Append a new test class to `tests/test_api.py` (after `ReductionApiTests`, before `CompletionStatusTests`):

```python
class MultiCoreSwitchTests(unittest.TestCase):
    """M-2: ``cores`` resolves to ``"auto"`` (max) by default, or an explicit int."""

    def setUp(self):
        self.columns = np.array(
            [[10, 1, 0], [0, 10, 1], [0, 0, 1]], dtype=np.int64
        )

    def test_resolve_cores_auto_detects_cpu_count(self):
        import os as _os
        from blaster.blaster import _resolve_cores

        detected = _os.cpu_count()
        expected = detected if (detected is not None and detected >= 1) else 1
        for spelling in ("auto", "Auto", "AUTO", " auto "):
            with self.subTest(spelling=spelling):
                self.assertEqual(_resolve_cores(spelling), expected)

    def test_resolve_cores_explicit_integer(self):
        from blaster.blaster import _resolve_cores

        self.assertEqual(_resolve_cores(1), 1)
        self.assertEqual(_resolve_cores(4), 4)

    def test_resolve_cores_rejects_invalid_values(self):
        from blaster.blaster import _resolve_cores

        for bad in (0, True, "bogus", "4", 2.5, None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "cores"):
                    _resolve_cores(bad)

    def test_resolve_cores_falls_back_to_one_when_cpu_count_is_none(self):
        from unittest import mock
        import blaster.blaster as blaster_module

        with mock.patch.object(blaster_module.os, "cpu_count", return_value=None):
            self.assertEqual(blaster_module._resolve_cores("auto"), 1)

    def test_reduce_cores_auto_matches_single_threaded(self):
        t1, r1, _ = reduce(self.columns, lll_size=3, cores=1, use_seysen=False)
        ta, ra, _ = reduce(self.columns, lll_size=3, cores="auto", use_seysen=False)

        np.testing.assert_array_equal(r1, ra)
        np.testing.assert_array_equal(t1, ta)

    def test_reduce_defaults_to_auto_cores(self):
        import os as _os
        from unittest import mock
        import blaster.blaster as blaster_module

        detected = _os.cpu_count()
        expected = detected if (detected is not None and detected >= 1) else 1
        with mock.patch.object(blaster_module, "set_num_cores") as set_cores:
            reduce(self.columns, lll_size=3, use_seysen=False)
            set_cores.assert_called_once_with(expected)

    def test_reduce_basis_forwards_cores_auto(self):
        rows = self.columns.T.copy()
        reduced_1 = reduce_basis(rows, lll_size=3, cores=1, use_seysen=False)
        reduced_auto = reduce_basis(rows, lll_size=3, cores="auto", use_seysen=False)

        np.testing.assert_array_equal(reduced_1, reduced_auto)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_api.MultiCoreSwitchTests -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_cores' from 'blaster.blaster'` for the unit tests; `ValueError: cores must be an integer` for the `cores="auto"` end-to-end tests; the `test_reduce_defaults_to_auto_cores` mock asserts `set_num_cores` was called with `1` but expected `os.cpu_count()`.

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/blaster.py`:

3a. Add imports — `import os` becomes the first import, and add `from typing import Union`:

```python
import os
from functools import partial
...
from typing import Union
```

3b. After `_integer_parameter` (line 180), add:

```python
def _resolve_cores(cores):
    """Resolve ``cores`` to a concrete thread count: ``"auto"`` (any case)
    selects all available cores; anything else must be an integer >= 1."""
    if isinstance(cores, str):
        if cores.strip().lower() == "auto":
            detected = os.cpu_count()
            return detected if (detected is not None and detected >= 1) else 1
        raise ValueError("cores must be 'auto' or an integer")
    return _integer_parameter("cores", cores, 1)
```

3c. Replace line 240:

```python
    cores = _integer_parameter("cores", cores, 1)
```

with:

```python
    cores = _resolve_cores(cores)
```

3d. Replace the `reduce()` signature's `cores` default (line 289):

```python
        B, lll_size: int = 64, delta: float = 0.99, cores: int = 1, debug: bool = False,
```

with:

```python
        B, lll_size: int = 64, delta: float = 0.99, cores: Union[int, str] = "auto", debug: bool = False,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_api.MultiCoreSwitchTests -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass; the existing `{"cores": 0}` case in `test_rejects_invalid_reduction_parameters` still passes (the explicit-integer error path is preserved).

- [ ] **Step 6: Commit**

```bash
git add src/blaster/blaster.py tests/test_api.py
git commit -m "feat(M-2): add cores='auto' default to reduce()

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: CLI `--cores auto`

**Files:**
- Modify: `src/blaster/app.py` (imports; add `_cores_type`; `--cores` arg ~line 32; remove clamp ~line 112-116)
- Test: `tests/test_io_cli.py` (add tests to `CommandLineTests`)

**Interfaces:**
- Consumes: `reduce(..., cores=...)` from `blaster.blaster` (resolves `"auto"` per Task 1); `build_parser` for the default assertion.
- Produces: `_cores_type(value)` (argparse type callable, returns `"auto"` or `int`); `--cores` default `"auto"`.

- [ ] **Step 1: Write the failing tests**

Append to the `CommandLineTests` class in `tests/test_io_cli.py`:

```python
    def test_cli_parser_default_cores_is_auto(self):
        from blaster.app import build_parser
        self.assertEqual(build_parser().parse_args([]).cores, "auto")

    def test_cli_cores_auto_reduces(self):
        source = io.StringIO()
        write_lattice(self.columns, source)
        result = self.run_cli(
            "--no-seysen", "--cores", "auto", input_text=source.getvalue()
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        reduced = parse_lattice(result.stdout, qary_order=False)
        self.assertEqual(reduced.shape, self.columns.shape)

    def test_cli_cores_integer_reduces(self):
        source = io.StringIO()
        write_lattice(self.columns, source)
        result = self.run_cli(
            "--no-seysen", "--cores", "2", input_text=source.getvalue()
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_omitted_cores_defaults_to_auto(self):
        source = io.StringIO()
        write_lattice(self.columns, source)
        result = self.run_cli("--no-seysen", input_text=source.getvalue())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_invalid_cores_string(self):
        result = self.run_cli(
            "--no-seysen", "--cores", "bogus", input_text="[[1 0] [0 1]]"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cores", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_cores_below_minimum(self):
        result = self.run_cli(
            "--no-seysen", "--cores", "0", input_text="[[1 0] [0 1]]"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cores", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_io_cli.CommandLineTests.test_cli_parser_default_cores_is_auto tests.test_io_cli.CommandLineTests.test_cli_cores_auto_reduces tests.test_io_cli.CommandLineTests.test_cli_rejects_cores_below_minimum -v`
Expected: FAIL — the parser default is `max(1, cpu_count() // 2)`, not `"auto"`; `--cores auto` errors with `invalid int value: 'auto'` (returncode 2, expected 0); `--cores 0` is silently clamped to `1` (returncode 0, expected 2).

- [ ] **Step 3: Write the minimal implementation**

In `src/blaster/app.py`:

3a. Imports — remove `ceil` and the `cpu_count` import (both become unused once the clamp is gone):

```python
from math import log2
```
(delete `from math import ceil, log2` and `from multiprocessing import cpu_count`)

3b. Add the argparse type callable after the imports:

```python
def _cores_type(value):
    """argparse type for --cores: 'auto' (case-insensitive) or an integer."""
    if value.strip().lower() == "auto":
        return "auto"
    return int(value)
```

3c. Replace the `--cores` argument (lines 32-35):

```python
    parser.add_argument(
        "--cores", "-j", type=_cores_type, default="auto",
        help="number of worker threads, or 'auto' to use all cores (default)",
    )
```

3d. Remove the clamp in `_run` (lines 112-116) — the `rank` line and the `max(1, min(...))` assignment:

```python
    rank = basis.shape[1]
    args.cores = max(
        1,
        min(args.cores, ceil(rank / max(2, args.lll_size)), cpu_count() // 2),
    )
```

The rest of `_run` is unchanged: `cores=args.cores` now passes `"auto"` or an `int` straight through to `reduce()`, which resolves it per Task 1.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_io_cli.CommandLineTests -v`
Expected: PASS (all `CommandLineTests`, including the new six and the existing `--cores 1` tests).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/blaster/app.py tests/test_io_cli.py
git commit -m "feat(M-2): support --cores auto (default) in the CLI

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** §2 (library `_resolve_cores` + signature + wiring) → Task 1; §3 (CLI `--cores` + clamp removal) → Task 2; §4 (determinism, reference engine untouched) → Task 1 tests 5/7 and the existing `test_reference_engine_rejects_unsupported_parameters`; §6 (acceptance driver items 1-7) → mapped across the two tasks' test steps.
- **Placeholder scan:** none — every step has concrete code and a concrete command.
- **Type consistency:** `_resolve_cores` is defined (Task 1) before `_cores_type` relies on `reduce` resolving `"auto"` (Task 2); `build_parser` default `"auto"` matches `reduce(cores="auto")`.
- **Review Focus:** all five lines have a named test in the owning task.
