"""Fast deterministic correctness and timing baseline for BLASter."""

import argparse
from multiprocessing import cpu_count
from time import perf_counter

import numpy as np

from blaster import reduce
from blaster.size_reduction import is_weakly_lll_reduced
from blaster.stats import get_profile, rhf


def qary_basis(dimension, modulus, rng):
    split = dimension // 2
    rows = np.zeros((dimension, dimension), dtype=np.int64)
    rows[:split, :split] = np.identity(split, dtype=np.int64)
    rows[:split, split:] = rng.integers(0, modulus, size=(split, dimension - split))
    rows[split:, split:] = modulus * np.identity(dimension - split, dtype=np.int64)
    return np.ascontiguousarray(rows.T)


def random_unimodular_basis(dimension, rng):
    basis = np.identity(dimension, dtype=np.int64)
    for _ in range(4 * dimension):
        source, target = rng.choice(dimension, size=2, replace=False)
        basis[:, target] += int(rng.choice([-3, -2, 2, 3])) * basis[:, source]
    return basis


def run_case(kind, basis, cores, lll_size, delta, use_seysen):
    started = perf_counter()
    transform, reduced, time_profile = reduce(
        basis,
        lll_size=lll_size,
        delta=delta,
        cores=cores,
        use_seysen=use_seysen,
    )
    walltime = perf_counter() - started

    if not np.array_equal(basis @ transform, reduced):
        raise RuntimeError(f"{kind}: transformation invariant failed")
    if abs(round(np.linalg.det(transform))) != 1:
        raise RuntimeError(f"{kind}: transformation is not unimodular")
    if not is_weakly_lll_reduced(np.linalg.qr(reduced, mode="r"), delta):
        raise RuntimeError(f"{kind}: weak LLL condition failed")

    profile = get_profile(reduced)
    print(
        f"{kind},{basis.shape[1]},{cores},{lll_size},{int(use_seysen)},"
        f"{time_profile.num_iterations},{walltime:.6f},{rhf(profile):.6f},ok"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--cores", type=int, default=min(2, max(1, cpu_count() // 2)))
    parser.add_argument("--lll-size", type=int, default=32)
    parser.add_argument("--delta", type=float, default=0.99)
    parser.add_argument("--modulus", type=int, default=4093)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--seysen", action="store_true")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    print("kind,dimension,cores,lll_size,seysen,iterations,walltime_s,rhf,status")
    for dimension in args.dimensions:
        if dimension < 2:
            parser.error("dimensions must be at least 2")
        lll_size = min(args.lll_size, dimension)
        run_case(
            "random",
            random_unimodular_basis(dimension, rng),
            args.cores,
            lll_size,
            args.delta,
            args.seysen,
        )
        run_case(
            "qary",
            qary_basis(dimension, args.modulus, rng),
            args.cores,
            lll_size,
            args.delta,
            args.seysen,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
