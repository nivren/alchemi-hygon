# Hybrid Dispatch & Pitfalls: Syevd vs Syevj on Haiguang DCU

## The $N \le 200$ Fallback Hypothesis

The source project profiled
rocsolver_*syevd_strided_batched on Haiguang DCU BW200 with DTK 26.04 and
reported a latency cliff around $N \approx 200$:

| Dimension $N$ | Batch $B$ | syevd | syevj | Reported syevj/syevd comparison |
| :---: | :---: | :---: | :---: | :---: |
| 64 | 16 | 148.7 ms | 18.2 ms | 8.17x |
| 100 | 16 | 134.6 ms | 21.0 ms | 6.41x |
| 138 | 16 | 284.5 ms | 27.2 ms | 10.46x |
| 199 | 16 | 332.7 ms | 36.4 ms | 9.14x |
| 201 | 16 | 45.3 ms | 46.1 ms | 1.02x |
| 552 | 16 | 89.3 ms | 245.0 ms | syevd 2.7x faster |

The source analysis attributes the small-matrix behavior to a possible
host-LAPACK path involving LAPACKE_dstedc and device-host transfers. This is a
source-specific hypothesis that must be rechecked with the actual DTK build,
tracing, and workload before implementation.

## Hybrid Dispatching Hypothesis

An implementation may inspect $N$ and benchmark:

    if n <= threshold:
        rocsolver_dsyevj_strided_batched(...)
    else:
        rocsolver_dsyevd_strided_batched(...)

The threshold is not a repository-wide invariant until confirmed by a
reproducible probe. Record solver status, convergence information, numerical
residuals, device, driver/DTK version, and timing methodology.
