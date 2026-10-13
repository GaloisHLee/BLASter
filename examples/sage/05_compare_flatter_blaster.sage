"""Compare Flatter and BLASter CLI time and reduction quality on q-ary lattices."""

import argparse
import csv
from math import sqrt
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from statistics import median

import numpy as np

from blaster.lattice_io import read_lattice, write_lattice
from blaster.stats import get_profile, rhf, slope


def qary_basis(dimension, bits, seed):
    split = dimension // 2
    tail = dimension - split
    modulus = int(next_prime(2**bits))
    rng = np.random.default_rng(int(seed) + int(dimension))
    block = rng.integers(0, modulus, size=(split, tail), dtype=np.int64)

    rows = np.zeros((dimension, dimension), dtype=np.int64)
    rows[:split, :split] = np.identity(split, dtype=np.int64)
    rows[:split, split:] = block
    rows[split:, split:] = modulus * np.identity(tail, dtype=np.int64)
    return rows, block, modulus


def run_command(command, environment):
    started = perf_counter()
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    elapsed = perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return elapsed


def read_row_basis(path):
    columns = read_lattice(path, qary_order=False)
    return np.ascontiguousarray(columns.T)


def verify_qary_lattice(output_rows, input_rows, block, modulus):
    output = matrix(ZZ, output_rows.tolist())
    original = matrix(ZZ, input_rows.tolist())
    split = block.shape[0]
    left = output.matrix_from_columns(range(split))
    right = output.matrix_from_columns(range(split, output.ncols()))
    numerator = right - left * matrix(ZZ, block.tolist())

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


def quality_metrics(rows):
    columns = np.ascontiguousarray(rows.T)
    profile = get_profile(columns)
    first_norm = sqrt(sum(int(value)**2 for value in rows[0]))
    return {
        "rhf": float(rhf(profile)),
        "slope": float(slope(profile)),
        "b1_norm": first_norm,
    }


def compare_dimension(args, dimension):
    input_rows, block, modulus = qary_basis(dimension, args.bits, args.seed)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.cores)
    environment["VECLIB_MAXIMUM_THREADS"] = str(args.cores)

    with TemporaryDirectory(prefix="flatter-blaster-") as directory:
        directory = Path(directory)
        input_path = directory / "input.lat"
        output_paths = {
            "blaster": directory / "blaster.lat",
            "flatter": directory / "flatter.lat",
        }
        write_lattice(np.ascontiguousarray(input_rows.T), input_path)

        commands = {
            "blaster": [
                shutil.which("blaster"),
                "--input", str(input_path),
                "--output", str(output_paths["blaster"]),
                "--cores", str(args.cores),
            ],
            "flatter": [
                shutil.which("flatter"),
                "-rhf", str(args.target_rhf),
                str(input_path),
                str(output_paths["flatter"]),
            ],
        }

        timings = {"blaster": [], "flatter": []}
        names = ["blaster", "flatter"]
        for trial in range(args.repeats):
            order = names if trial % 2 == 0 else list(reversed(names))
            for name in order:
                timings[name].append(run_command(commands[name], environment))

        rows = []
        for name in names:
            output_rows = read_row_basis(output_paths[name])
            verify_qary_lattice(output_rows, input_rows, block, modulus)
            metrics = quality_metrics(output_rows)
            rows.append({
                "dimension": dimension,
                "bits": args.bits,
                "cores": args.cores,
                "repeats": args.repeats,
                "tool": name,
                "median_seconds": median(timings[name]),
                "rhf": metrics["rhf"],
                "slope": metrics["slope"],
                "b1_norm": metrics["b1_norm"],
                "verified": "yes",
            })
        return rows


def print_results(results):
    fields = [
        "dimension", "bits", "cores", "repeats", "tool",
        "median_seconds", "rhf", "slope", "b1_norm", "verified",
    ]
    print(",".join(fields))
    for row in results:
        print(
            f"{row['dimension']},{row['bits']},{row['cores']},{row['repeats']},"
            f"{row['tool']},{row['median_seconds']:.6f},{row['rhf']:.6f},"
            f"{row['slope']:.6f},{row['b1_norm']:.6f},{row['verified']}"
        )

    print("\nPer-dimension summary:")
    for dimension in sorted({row["dimension"] for row in results}):
        pair = {row["tool"]: row for row in results if row["dimension"] == dimension}
        faster = min(pair, key=lambda name: pair[name]["median_seconds"])
        ratio = max(row["median_seconds"] for row in pair.values()) / min(
            row["median_seconds"] for row in pair.values()
        )
        better = min(pair, key=lambda name: pair[name]["rhf"])
        quality_gap = abs(pair["blaster"]["rhf"] - pair["flatter"]["rhf"])
        print(
            f"n={dimension}: faster={faster} ({ratio:.2f}x spread), "
            f"lower_rhf={better}, rhf_gap={quality_gap:.6f}"
        )


def write_csv(path, results):
    fields = [
        "dimension", "bits", "cores", "repeats", "tool",
        "median_seconds", "rhf", "slope", "b1_norm", "verified",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--bits", type=int, default=20)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--target-rhf", type=float, default=1.0219)
    parser.add_argument("--csv")
    args = parser.parse_args()

    args.dimensions = [int(dimension) for dimension in args.dimensions]
    args.bits = int(args.bits)
    args.cores = int(args.cores)
    args.repeats = int(args.repeats)
    args.seed = int(args.seed)
    args.target_rhf = float(args.target_rhf)

    for executable in ("blaster", "flatter"):
        if shutil.which(executable) is None:
            parser.error(f"{executable!r} is not available on PATH")
    if any(dimension < 4 or dimension % 2 for dimension in args.dimensions):
        parser.error("dimensions must be even integers of at least 4")
    if not 1 <= args.bits <= 30:
        parser.error("bits must be between 1 and 30 for the shared int64 test domain")
    if args.cores < 1 or args.repeats < 1:
        parser.error("cores and repeats must be positive")

    results = []
    for dimension in args.dimensions:
        results.extend(compare_dimension(args, dimension))

    print_results(results)
    if args.csv:
        write_csv(args.csv, results)
    print("FLATTER_BLASTER_COMPARE_OK")


main()
