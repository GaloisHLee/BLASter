# Sage examples

These examples use the BLASter package installed in the Conda environment
`sage106`. From the repository root, build and install the current source first:

```console
make install
```

Run one example:

```console
conda run --no-capture-output -n sage106 sage examples/sage/01_basic_lll.sage
```

Run every example:

```console
bash examples/sage/run_all.sh
```

Examples 05 and 06 additionally require `flatter` on `PATH`.

The examples cover:

- `01_basic_lll.sage`: row-oriented LLL with classical size reduction;
- `02_deep_lll.sage`: DeepLLL-2 with Seysen reduction;
- `03_bkz_qary.sage`: one BKZ-3 tour on a small q-ary lattice;
- `04_column_basis.sage`: a rectangular basis using BLASter's column convention;
- `05_compare_flatter_blaster.sage`: CLI-to-CLI timing and quality comparison;
  and
- `06_large_integer_boundary.sage`: observed behavior beyond current BLASter's
  int64 domain.

The reduction examples check their transformation identity, unimodularity, weak
LLL condition, and first-vector norm. The comparison examples verify exact
lattice equivalence and the expected int64 boundary. Every script prints an
`*_OK` marker after its checks pass.

Run a larger Flatter/BLASter comparison and save machine-readable results:

```console
conda run --no-capture-output -n sage106 sage \
  examples/sage/05_compare_flatter_blaster.sage \
  --dimensions 64 128 256 512 --cores 2 --repeats 3 \
  --csv benchmark/flatter_vs_blaster.csv
```

Research interpretation, algorithm-separation rules, and reproducibility limits
are recorded in
[`benchmark/FLATTER_VS_BLASTER.md`](../../benchmark/FLATTER_VS_BLASTER.md).
