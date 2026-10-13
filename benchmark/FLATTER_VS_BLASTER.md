# Flatter and BLASter research comparison

This note records a narrow comparison between two different lattice-reduction
algorithms. It does not define Flatter as a BLASter arithmetic backend, runtime
fallback, preprocessing stage, or correctness oracle.

## Interpretation boundary

BLASter combines segmented local LLL/DeepLLL, divide-and-conquer classical or
Seysen panel reduction, and an experimental segmented BKZ path. Flatter has its
own recursive matrix-based algorithm, precision policy, controls, and stopping
semantics. Matching input syntax does not make their trajectories or output
contracts interchangeable.

The comparison therefore asks only empirical questions on a declared corpus:

- how long did each complete under its stated parameters;
- what output quality metrics were observed; and
- did an exact, corpus-specific lattice-equivalence check accept the output?

It does not answer which algorithm is universally better or which should be
silently selected for a new input.

## Common-input benchmark

Command:

```console
conda run --no-capture-output -n sage106 sage \
  examples/sage/05_compare_flatter_blaster.sage \
  --dimensions 64 128 256 512 --bits 20 --cores 2 --repeats 3 \
  --csv benchmark/flatter_vs_blaster.csv
```

Both tools received the same deterministic fplll-format q-ary basis. Flatter
used an explicit RHF target of 1.0219. BLASter used its default LLL delta of 0.99
with Seysen reduction. These parameters are native controls, not evidence that
the algorithms had identical stopping targets. Tool order alternated between
trials; each table time is the median of three CLI runs.

The script reconstructs a transformation from exact q-ary coordinates and
checks it is integer unimodular. That check is useful evidence for this corpus;
it is not yet the general independent verifier specified for future BLASter.

| Dimension | BLASter median | Flatter median | Observed time ratio | BLASter RHF | Flatter RHF |
|---:|---:|---:|---:|---:|---:|
| 64  | 0.106 s | 0.433 s | BLASter 4.07x lower | 1.015968 | 1.018915 |
| 128 | 0.197 s | 2.947 s | BLASter 14.98x lower | 1.019934 | 1.019749 |
| 256 | 0.667 s | 23.197 s | BLASter 34.79x lower | 1.021682 | 1.019411 |
| 512 | 4.564 s | 101.295 s | BLASter 22.19x lower | 1.013630 | 1.013630 |

Lower RHF is better. Neither algorithm had lower RHF at every dimension:
BLASter was lower at 64, Flatter was lower at 128 and 256, and the printed values
matched at 512. Time and quality must remain separate observations.

Raw results are stored in [`flatter_vs_blaster.csv`](flatter_vs_blaster.csv).

## Arithmetic-domain observation

`examples/sage/06_large_integer_boundary.sage` constructs a dimension-16 q-ary
basis with 129-bit entries. On the recorded machine:

```text
flatter_seconds=0.092311, verified=yes
blaster_status=rejected_as_expected (outside signed int64)
```

Flatter completed and its q-ary lattice relation was checked exactly. Current
BLASter rejected the input before reduction because its Cython/Eigen core stores
integer matrices in signed int64.

This establishes a limitation of the current BLASter implementation, not a
runtime-routing rule. The planned CheckedInt64 and Multiprecision strategies
must implement BLASter's own specified algorithm. They must not obtain wider
integer support by invoking Flatter.

## Semantic comparison

| Research dimension | BLASter in this worktree | Flatter in this experiment |
|---|---|---|
| Algorithm identity | segmented BLASter control flow | Flatter's independent recursive reducer |
| Integer domain | signed int64 input/state, unchecked intermediates | supports the 129-bit boundary sample |
| Primary public controls | delta, segment size, Seysen, DeepLLL, experimental BKZ | alpha/RHF/delta/log-condition CLI targets |
| Python/Sage transform | returned directly by `reduce`/`reduce_basis` | comparison script reconstructs it for q-ary inputs |
| Completion evidence here | BLASter weak-LLL and transform assertions | process completion, quality metrics, corpus-specific exact relation |
| Role in this project | algorithm under specification and future arithmetic work | external benchmark and research comparison |

fpylll/fplll has the same project role as Flatter: it can provide generators,
interoperability cases, and independent experimental samples, but its traditional
LLL/BKZ interfaces are not BLASter backends or specification truth.

## Responsible use of the data

The measurements support hypotheses worth testing, not a selector:

- BLASter's segmented approach was fast on this low-bit q-ary corpus.
- Quality ordering changed with dimension even on one corpus.
- Current BLASter's integer representation excludes a domain that Flatter can
  process.
- Dimension, bit width, structure, native stopping parameters, thread count,
  hardware, and verification method all need to be reported for a comparison.

Future experiments should add corpus families and repeated distributions rather
than convert these four points into a hard-coded `if` statement. A workflow that
explicitly runs several algorithms for research must report each stage and its
transformation separately; it is not one multi-backend BLASter run.

## Reproducibility

The baseline was recorded on 2026-08-08 on Apple arm64 with two threads,
SageMath 10.6, Python 3.11.13, the pre-sync BLASter 0.1.0 working tree, and
`/usr/local/bin/flatter`. The local Flatter source checkout was at `13c4ef0`.
The current checkout has since synchronized upstream `68225ec` and reports
package version 1.0, so this table is historical comparison evidence rather
than a claim about the post-sync binary.

The comparison is intentionally limited to deterministic 20-bit q-ary bases.
Re-run it for every workload claim, preserving both native parameter sets and
reporting quality independently of time.

Flatter's design and controls are documented in its
[official repository](https://github.com/keeganryan/flatter). BLASter's frozen
algorithm semantics and arithmetic direction are in
[`docs/spec/blaster-algorithm.md`](../docs/spec/blaster-algorithm.md) and
[`docs/spec/arithmetic-model.md`](../docs/spec/arithmetic-model.md).
