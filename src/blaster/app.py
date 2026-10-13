"""Command-line interface for BLASter lattice reduction."""

import argparse
from math import ceil, log2
from multiprocessing import cpu_count
from pathlib import Path
from sys import stderr

import numpy as np

from .blaster import reduce
from .lattice_io import read_qary_lattice, write_lattice
from .stats import gaussian_heuristic, get_profile, rhf, slope


def build_parser():
    parser = argparse.ArgumentParser(
        prog="BLASter",
        description="Reduce an integer lattice basis with segmented LLL, DeepLLL, or BKZ",
        epilog="Input and output use fplll matrix format.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="show reduction progress")
    parser.add_argument(
        "--cores", "-j", type=int, default=max(1, cpu_count() // 2),
        help="number of worker threads",
    )

    parser.add_argument("--input", "-i", help="input file (default: stdin)")
    parser.add_argument("--output", "-o", help="output file (default: stdout)")
    parser.add_argument("--logfile", "-l", help="write reduction metrics as CSV")
    parser.add_argument(
        "--profile", "-p", action="store_true", dest="debug",
        help="validate local reductions and print the final profile",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="do not print the basis")
    parser.add_argument("--anim", "-a", help="write a GIF of the evolving basis profile")

    parser.add_argument("--delta", type=float, default=0.99, help="Lovasz delta factor")
    parser.add_argument(
        "--lll-size", "--lll_size", "-L", type=int, default=64, dest="lll_size",
        help="local LLL/DeepLLL segment size",
    )
    parser.add_argument(
        "--no-seysen", "-s", action="store_false", dest="use_seysen",
        help="use size reduction instead of Seysen reduction",
    )
    parser.set_defaults(use_seysen=True)
    parser.add_argument(
        "--depth", "-d", type=int, default=0,
        help="maximum DeepLLL insertion depth",
    )

    parser.add_argument("--beta", "-b", type=int, help="BKZ block size (maximum: 80)")
    parser.add_argument("--bkz-tours", "-t", type=int, default=8, help="number of BKZ tours")
    parser.add_argument(
        "--bkz-size", "-S", type=int,
        help="parallel BKZ segment size (default: max of LLL size and beta)",
    )
    parser.add_argument(
        "--bkz-prog", "-P", type=int,
        help="progressive BKZ block-size increment",
    )
    return parser


def _print_expected_quality(basis):
    rank = basis.shape[1]
    log_slope = log2(1.02190)
    profile = get_profile(basis)
    log_det = sum(profile)
    norm_b1 = 2.0 ** (log_slope * (rank - 1) + log_det / rank)

    comparison = ""
    if np.count_nonzero(basis[:, 0]) == 1:
        q = sum(basis[:, 0])
        operator = "<" if norm_b1 < q else ">="
        comparison = f"{operator} {int(q):d} "
    print(
        f"E[||b_1||] ~ {norm_b1:.2f} {comparison}"
        f"(GH: lambda_1 ~ {gaussian_heuristic(profile):.2f})",
        file=stderr,
    )


def _run(args):
    if args.input and args.output and Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("input and output files must be different")

    basis = read_qary_lattice(args.input)
    rank = basis.shape[1]
    args.cores = max(
        1,
        min(args.cores, ceil(rank / max(2, args.lll_size)), cpu_count() // 2),
    )

    if args.verbose:
        _print_expected_quality(basis)

    transform, reduced, time_profile = reduce(
        basis,
        lll_size=args.lll_size,
        delta=args.delta,
        cores=args.cores,
        debug=args.debug,
        verbose=args.verbose,
        logfile=args.logfile,
        anim=args.anim,
        depth=args.depth,
        use_seysen=args.use_seysen,
        beta=args.beta,
        bkz_tours=args.bkz_tours,
        bkz_size=args.bkz_size,
        bkz_prog=args.bkz_prog,
    )

    if args.output:
        write_lattice(reduced, args.output)
    elif not args.quiet:
        write_lattice(reduced)

    if args.verbose:
        print(f"\n{time_profile}", file=stderr)
    if args.debug:
        profile = get_profile(reduced)
        print(
            "\nProfile = [" + " ".join(f"{value:.2f}" for value in profile) + "]\n"
            f"RHF = {rhf(profile):.5f}^n, slope = {slope(profile):.6f}, "
            f"||b_1|| = {2.0**profile[0]:.1f}",
            file=stderr,
        )

    if not np.array_equal(basis @ transform, reduced):
        raise RuntimeError("internal error: transformation does not reproduce the reduced basis")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
