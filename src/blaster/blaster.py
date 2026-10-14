"""
BLASter lattice reduction: LLL with QR decomposition, Seysen's reduction, and
segments, in which lattice reduction is done in parallel.
"""
from functools import partial
from math import isfinite
from numbers import Integral, Real
from sys import stderr
from time import perf_counter_ns

import numpy as np

from ._matrix import as_int64_matrix
from ._core import set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, \
    ZZ_right_matmul
from .size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce
from .stats import get_profile, rhf, slope, potential


MAX_BKZ_BLOCK_SIZE = 80


class TimeProfile:
    """
    Object containing time spent on different parts when running BLASter.
    """

    def __init__(self, use_seysen: bool = False):
        self._strs = [
            "QR-decomp.", "LLL-red.", "BKZ-red.",
            "Seysen-red." if use_seysen else "Size-red.  ", "Matrix-mul."
        ]
        self.num_iterations = 0
        self.times = [0] * 5
        self.interrupted = False  # True when reduction was cut short by KeyboardInterrupt.

    def tick(self, *times):
        self.num_iterations += 1
        self.times = [x + y for x, y in zip(self.times, times)]

    def __str__(self):
        return (
            f"Iterations: {self.num_iterations}\n" +
            "\n".join(f"t_{{{s:11}}}={t/10**9:10.3f}s" for s, t in zip(self._strs, self.times) if t)
        )


def lll_reduce(B, U, U_seysen, lll_size, delta, depth,
               tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's lattice reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, use deep insertions up to depth `depth`.
    """
    n, is_reduced, offset = B.shape[1], False, 0
    red_fn = partial(block_deep_lll, depth) if depth else block_lll

    # Keep running until the basis is LLL reduced.
    while not is_reduced:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call LLL concurrently on small blocks.
        t2 = perf_counter_ns()
        offset = lll_size // 2 if offset == 0 else 0
        red_fn(R, B, U, delta, offset, lll_size)  # LLL or deep-LLL

        if debug:
            for i in range(offset, n, lll_size):
                j = min(n, i + lll_size)
                # Check whether R_[i:j) is really LLL-reduced.
                assert is_lll_reduced(R[i:j, i:j], delta)

        # Step 3: QR-decompose again because LLL "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen's reduction.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        # Step 6: Check whether the basis is weakly-LLL reduced.
        t6 = perf_counter_ns()

        is_reduced = is_weakly_lll_reduced(R, delta)
        tprof.tick(t2 - t1 + t4 - t3, t3 - t2, 0, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"DeepLLL-{depth}" if depth else "LLL", None)
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)


def bkz_reduce(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, bkz_size, tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's BKZ reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, BLASter's deep-LLL is called in between calls of the SVP oracle.
    Otherwise BLASter's LLL is run.
    """
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0

    lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)

    while tours_done < bkz_tours:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call BKZ concurrently on small blocks!
        t2 = perf_counter_ns()
        # norm_before = abs(R[cur_front, cur_front])
        block_bkz(beta, R, B, U, delta, cur_front % beta, bkz_size)

        # Step 3: QR-decompose again because BKZ "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')
        # assert abs(R[cur_front, cur_front]) <= norm_before

        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen's reduction.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, 0, t3 - t2, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"BKZ-{beta}", (beta, tours_done, bkz_tours, cur_front))
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)

        # After printing: update the current location of the 'reduction front'
        if cur_front + beta > n:
            # HKZ-reduction was performed at the end, which is the end of a tour.
            cur_front = 0
            tours_done += 1
        else:
            cur_front += (bkz_size - beta + 1)

        # Perform a final LLL reduction at the end
        lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)


def _integer_parameter(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


# Primes < 2**31 keep every intermediate product below 2**62, so the modular
# Gaussian elimination in `_rank_mod_p` cannot overflow int64.
_RANK_PRIMES = (2147483647, 2147483629, 2147483587, 2147483579, 2147483563)


def _rank_mod_p(B, p):
    """Return the exact rank of the integer matrix ``B`` over GF(p)."""
    A = np.mod(B, p).astype(np.int64)
    m, n = A.shape
    r = 0
    for col in range(n):
        pivot = r
        while pivot < m and A[pivot, col] == 0:
            pivot += 1
        if pivot == m:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        inv = pow(int(A[r, col]), p - 2, p)
        A[r] = (A[r] * inv) % p
        for row in range(r + 1, m):
            if A[row, col]:
                A[row] = (A[row] - A[row, col] * A[r]) % p
        r += 1
        if r == m:
            break
    return r


def _is_full_column_rank(B):
    """Return whether the columns of the integer matrix ``B`` are Q-linearly independent."""
    n = B.shape[1]
    return any(_rank_mod_p(B, p) == n for p in _RANK_PRIMES)


def _validated_parameters(
        B, lll_size, delta, cores, depth, beta, bkz_tours, bkz_size, bkz_prog):
    B = as_int64_matrix(B, name="basis")
    ambient_dimension, rank = B.shape
    if ambient_dimension < rank:
        raise ValueError("column basis rank cannot exceed its ambient dimension")
    # A float64 SVD rank test is unreliable here: NumPy's default tolerance is
    # s_max * max(M, N) * eps, so a full-rank integer basis whose columns span a
    # wide dynamic range (e.g. O(1) and O(2**60), as in HNP lattices) is misread
    # as rank deficient.  Rank over GF(p) never exceeds rank over Q, so a
    # full-rank reading modulo any one prime proves independence, and checking
    # several primes makes a spurious "dependent" verdict vanishingly unlikely.
    if not _is_full_column_rank(B):
        raise ValueError("basis vectors must be linearly independent")

    if isinstance(delta, bool) or not isinstance(delta, Real) or not isfinite(float(delta)):
        raise ValueError("delta must be a finite real number")
    delta = float(delta)
    if not 0.25 < delta < 1.0:
        raise ValueError("delta must satisfy 0.25 < delta < 1.0")

    lll_size = _integer_parameter("lll_size", lll_size, 2)
    cores = _integer_parameter("cores", cores, 1)
    depth = _integer_parameter("depth", depth, 0)
    effective_lll_size = min(lll_size, rank)

    if beta is None:
        return (
            B, effective_lll_size, delta, cores, depth,
            None, None, None, None,
        )

    beta = _integer_parameter("beta", beta, 2)
    if beta > MAX_BKZ_BLOCK_SIZE:
        raise ValueError(f"beta must not exceed {MAX_BKZ_BLOCK_SIZE}")
    if beta > rank:
        raise ValueError("beta must not exceed the basis rank")
    if depth:
        raise ValueError("depth cannot be combined with beta")

    bkz_tours = _integer_parameter("bkz_tours", bkz_tours, 1)
    if bkz_size is None:
        bkz_size = max(effective_lll_size, beta)
    bkz_size = _integer_parameter("bkz_size", bkz_size, 2)
    if bkz_size < beta:
        raise ValueError("bkz_size must be at least beta")
    bkz_size = min(bkz_size, rank)

    if bkz_prog is None:
        bkz_prog = beta
    bkz_prog = _integer_parameter("bkz_prog", bkz_prog, 1)
    if bkz_prog > beta:
        raise ValueError("bkz_prog must not exceed beta")

    return (
        B, effective_lll_size, delta, cores, depth,
        beta, bkz_tours, bkz_size, bkz_prog,
    )


def _progressive_betas(beta, increment):
    """Return the upstream progressive-beta schedule for the validated block size."""
    if beta < 40:
        return ()
    if increment >= beta:
        return (beta,)
    start = 40 + ((beta - 40) % increment)
    return tuple(range(start, beta + 1, increment))


def reduce(
        B, lll_size: int = 64, delta: float = 0.99, cores: int = 1, debug: bool = False,
        verbose: bool = False, logfile=None, anim=None, depth: int = 0,
        use_seysen: bool = False, beta=None, bkz_tours: int = 8,
        bkz_size=None, bkz_prog=None):
    """
    Reduce an int64 basis whose vectors are stored as columns.

    ``delta`` is the Lovasz factor. ``lll_size`` and ``bkz_size`` control the
    parallel segment sizes, while ``cores`` controls the OpenMP worker count.
    Set ``depth`` for DeepLLL, or set ``beta`` (up to 80) for BKZ. DeepLLL and
    BKZ cannot be requested together. ``use_seysen=False`` selects classical
    size reduction instead of Seysen reduction.

    For compatibility with the upstream algorithm, ``beta < 40`` selects
    DeepLLL-4 preprocessing rather than a pruned BKZ tour.

    :return: tuple (U, B · U, tprof) where:
        U: the transformation matrix such that B · U is LLL reduced,
        B · U: an LLL-reduced basis,
        tprof: TimeProfile object.
    """
    (
        B, lll_size, delta, cores, depth,
        beta, bkz_tours, bkz_size, bkz_prog,
    ) = _validated_parameters(
        B, lll_size, delta, cores, depth, beta, bkz_tours, bkz_size, bkz_prog
    )
    n, tprof = B.shape[1], TimeProfile(use_seysen)

    set_num_cores(cores)
    set_debug_flag(1 if debug else 0)

    tracers = {}
    if verbose:
        def trace_print(_, prof, note):
            log_str = '.'
            if note[0].startswith('BKZ'):
                beta, tour, ntours, touridx = note[1]
                log_str = (f"\nBKZ(β:{beta:3d},t:{tour + 1:2d}/{ntours:2d}, o:{touridx:4d}): "
                           f"slope={slope(prof):.6f}, rhf={rhf(prof):.6f}")
            print(log_str, end="", file=stderr, flush=True)
        tracers['v'] = trace_print

    # Set up logfile
    has_logfile = logfile is not None
    if has_logfile:
        tstart = perf_counter_ns()
        logfile_handle = open(logfile, "w", encoding="utf8")
        print('it,walltime,rhf,slope,potential,note', file=logfile_handle, flush=True)

        def trace_logfile(it, prof, note):
            walltime = (perf_counter_ns() - tstart) * 10**-9
            print(f'{it:4d},{walltime:.6f},{rhf(prof):8.6f},{slope(prof):9.6f},'
                  f'{potential(prof):9.3f},{note[0]}', file=logfile_handle, flush=True)

        tracers['l'] = trace_logfile

    # Set up animation lazily so matplotlib remains an optional dependency.
    has_animation = anim is not None
    if has_animation:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import ArtistAnimation, PillowWriter
        except ImportError as exc:
            raise RuntimeError(
                "animation output requires the 'animation' optional dependencies"
            ) from exc
        fig, ax = plt.subplots()
        ax.set(xlim=[0, n])
        artists = []

        def trace_anim(_, prof, __):
            artists.append(ax.plot(range(n), prof, color="blue"))

        tracers['a'] = trace_anim

    B = B.copy()  # Do not modify B in-place, but work with a copy.
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)

    try:
        try:
            if beta is None:
                lll_reduce(
                    B, U, U_seysen, lll_size, delta, depth,
                    tprof, tracers, debug, use_seysen,
                )
            elif beta < 40:
                # Upstream behavior for small requested block sizes: run
                # DeepLLL-4 without starting a pruned BKZ tour.
                lll_reduce(
                    B, U, U_seysen, lll_size, delta, 4,
                    tprof, tracers, debug, use_seysen,
                )
            else:
                # DeepLLL-4 is the preprocessing strategy used by the original BKZ path.
                for beta_ in _progressive_betas(beta, bkz_prog):
                    bkz_reduce(
                        B, U, U_seysen, lll_size, delta, 4, beta_,
                        bkz_tours if beta_ == beta else 1, bkz_size,
                        tprof, tracers, debug, use_seysen,
                    )
        except KeyboardInterrupt:
            tprof.interrupted = True  # Preserve the original behavior and return the partially reduced basis.
    finally:
        if has_logfile:
            logfile_handle.close()

    # Save and/or show the animation
    if has_animation:
        # Saving the animation takes a LONG time.
        if verbose:
            print('\nOutputting animation...', file=stderr)
        fig.tight_layout()
        ani = ArtistAnimation(fig=fig, artists=artists, interval=200)
        # Generate 1920x1080 image:
        plt.gcf().set_size_inches(16, 9)
        # plt.show()
        ani.save(anim, dpi=120, writer=PillowWriter(fps=5))

    return U, B, tprof
