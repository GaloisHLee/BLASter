"""Observe Flatter and current BLASter at a 129-bit integer-domain boundary."""

from math import sqrt
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from blaster.stats import get_profile, rhf


TOKEN = re.compile(r"\[|\]|[+-]?\d+")


def write_fplll(basis, path):
    rows = ["[" + " ".join(str(value) for value in row) + "]" for row in basis.rows()]
    Path(path).write_text("[" + "\n".join(rows) + "]\n", encoding="utf-8")


def read_fplll(path):
    text = Path(path).read_text(encoding="utf-8")
    tokens = []
    position = 0
    for match in TOKEN.finditer(text):
        assert not text[position:match.start()].strip()
        tokens.append(match.group())
        position = match.end()
    assert not text[position:].strip()

    cursor = 0

    def take(expected=None):
        nonlocal cursor
        assert cursor < len(tokens)
        token = tokens[cursor]
        if expected is not None:
            assert token == expected
        cursor += 1
        return token

    take("[")
    rows = []
    while tokens[cursor] == "[":
        take("[")
        row = []
        while tokens[cursor] != "]":
            row.append(ZZ(take()))
        take("]")
        rows.append(row)
    take("]")
    assert cursor == len(tokens)
    return matrix(ZZ, rows)


def qary_basis(dimension, bits):
    split = dimension // 2
    tail = dimension - split
    modulus = next_prime(2**bits)
    block = matrix(
        ZZ,
        split,
        tail,
        lambda row, column: (17 * row + 31 * column + 11) % modulus,
    )
    basis = block_matrix(ZZ, [
        [identity_matrix(ZZ, split), block],
        [zero_matrix(ZZ, tail, split), modulus * identity_matrix(ZZ, tail)],
    ])
    return basis, block, modulus


def verify_qary_lattice(output, original, block, modulus):
    split = block.nrows()
    left = output.matrix_from_columns(range(split))
    right = output.matrix_from_columns(range(split, output.ncols()))
    numerator = right - left * block
    assert all(value % modulus == 0 for value in numerator.list())
    tail_coefficients = matrix(
        ZZ,
        numerator.nrows(),
        numerator.ncols(),
        [value // modulus for value in numerator.list()],
    )
    transform = left.augment(tail_coefficients)
    assert transform * original == output
    assert abs(transform.det()) == 1


dimension = 16
bits = 128
basis, block, modulus = qary_basis(dimension, bits)
assert max(abs(value) for value in basis.list()).nbits() > 64
assert shutil.which("flatter") is not None
assert shutil.which("blaster") is not None

environment = os.environ.copy()
environment["OMP_NUM_THREADS"] = "1"

with TemporaryDirectory(prefix="large-integer-boundary-") as directory:
    directory = Path(directory)
    input_path = directory / "input.lat"
    flatter_path = directory / "flatter.lat"
    blaster_path = directory / "blaster.lat"
    write_fplll(basis, input_path)

    started = perf_counter()
    flatter_result = subprocess.run(
        [shutil.which("flatter"), str(input_path), str(flatter_path)],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    flatter_seconds = perf_counter() - started
    assert flatter_result.returncode == 0, flatter_result.stderr

    blaster_result = subprocess.run(
        [
            shutil.which("blaster"),
            "--input", str(input_path),
            "--output", str(blaster_path),
            "--cores", "1",
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert blaster_result.returncode == 2
    assert "int64" in blaster_result.stderr

    reduced = read_fplll(flatter_path)
    verify_qary_lattice(reduced, basis, block, modulus)
    columns = np.asarray(
        [float(value) for value in reduced.list()], dtype=np.float64
    ).reshape(reduced.dimensions()).T
    profile = get_profile(columns)
    first_norm = sqrt(sum(int(value)**2 for value in reduced.row(0)))

print(f"dimension={dimension}, largest_input_bits={modulus.nbits()}")
print(f"flatter_seconds={flatter_seconds:.6f}, rhf={rhf(profile):.6f}, "
      f"b1_norm={first_norm:.6f}, verified=yes")
print("blaster_status=rejected_as_expected (outside signed int64)")
print("LARGE_INTEGER_BOUNDARY_OK")
