# Smoke benchmark baseline

Recorded on 2026-08-08 with:

- macOS 15.7.9 on Apple arm64;
- Conda environment `sage106`;
- SageMath 10.6 / Python 3.11.13;
- NumPy 2.3.0 and Cython 3.1.4; and
- two BLASter worker threads with LLL segment size 32.

Commands:

```console
conda run -n sage106 python benchmark/smoke_benchmark.py --dimensions 32 64 --cores 2 --lll-size 32
conda run -n sage106 python benchmark/smoke_benchmark.py --dimensions 32 64 --cores 2 --lll-size 32 --seysen
```

Classical size reduction:

```text
kind,dimension,cores,lll_size,seysen,iterations,walltime_s,rhf,status
random,32,2,32,0,2,0.001722,1.000000,ok
qary,32,2,32,0,2,0.001943,1.014253,ok
random,64,2,32,0,5,0.006050,1.000000,ok
qary,64,2,32,0,12,0.016387,1.020563,ok
```

Seysen reduction:

```text
kind,dimension,cores,lll_size,seysen,iterations,walltime_s,rhf,status
random,32,2,32,1,2,0.002266,1.000000,ok
qary,32,2,32,1,2,0.001498,1.014253,ok
random,64,2,32,1,5,0.003898,1.000000,ok
qary,64,2,32,1,12,0.011529,1.020563,ok
```

Every row marked `ok` passed all of the following checks:

- `original @ transform == reduced`;
- the transformation determinant has absolute value one; and
- the reduced QR basis satisfies BLASter's weak LLL condition.

Wall time is a smoke baseline rather than a stable performance guarantee.

## Post-upstream-sync re-run

On 2026-08-13, after synchronizing upstream `68225ec` and rebuilding
`blaster._core`, the same commands were rerun with `PYTHONPATH=src`:

```text
kind,dimension,cores,lll_size,seysen,iterations,walltime_s,rhf,status
random,32,2,32,0,2,0.002259,1.000000,ok
qary,32,2,32,0,2,0.002864,1.014253,ok
random,64,2,32,0,5,0.010166,1.000000,ok
qary,64,2,32,0,12,0.028813,1.020563,ok
```

```text
kind,dimension,cores,lll_size,seysen,iterations,walltime_s,rhf,status
random,32,2,32,1,2,0.001615,1.000000,ok
qary,32,2,32,1,2,0.002777,1.014253,ok
random,64,2,32,1,5,0.007232,1.000000,ok
qary,64,2,32,1,12,0.019560,1.020563,ok
```

Iteration counts and RHF values match the pre-sync run on this corpus.
