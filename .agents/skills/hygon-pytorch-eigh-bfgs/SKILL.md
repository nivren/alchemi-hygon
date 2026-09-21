---
name: hygon-pytorch-eigh-bfgs
description: Develop, optimize, and validate batched symmetric eigensolvers (eigh/evd), BFGS Quasi-Newton optimizers, and spectral Hessian conditioning for PyTorch on Haiguang DCU / AMD ROCm. Use when diagnosing slow batched eigh, replacing PyTorch single-matrix loops, dispatching rocSOLVER strided-batched kernels (syevj vs syevd), or resolving memory layout, stream synchronization, and silent CPU fallback issues.
---

# Hygon DCU PyTorch Batched Eigh & BFGS Optimization

Guide for implementing, debugging, optimizing, and validating high-performance
batched symmetric eigensolvers (eigh/evd) and Quasi-Newton (BFGS/L-BFGS)
spectral optimization in PyTorch on Haiguang DCU (DTK 25.x / 26.x).

## Core Problem & Mechanism

1. **PyTorch Native GPU Bottleneck**: PyTorch's
   `torch.linalg.eigh` on GPU internally delegates to
   `BatchLinearAlgebraLib.cpp`, which loops sequentially over batch
   elements and calls single-matrix solvers.
2. **Native Solution**: Bypass `hipsolverDn` compatibility wrappers and
   call DTK's native `rocSOLVER` strided-batched kernels through PyTorch
   `TORCH_LIBRARY` C++ extensions.
3. **The Hidden $N \le 200$ Fallback Trap**:
   `rocsolver_*syevd_strided_batched` may contain an internal threshold.
   For $N \le 200$, verify whether it falls back to host CPU
   `LAPACKE_dstedc`, copying data across PCIe and executing sequentially.
4. **Hybrid Dispatch Candidate**:
   - $N \le 200$: consider `rocsolver_*syevj_strided_batched`.
   - $N > 200$: consider `rocsolver_*syevd_strided_batched`.

## Workflow

### 1. Classify the Problem Contract

Record:
- Matrix dimension $N$ (for crystal systems with $A$ atoms, often $N = 3A$).
- Batch size $B$.
- Precision: `torch.float64` or `torch.float32`.
- Algorithm needs: eigenvalues only or the full eigensystem.

### 2. Choose the Dispatch Strategy

Use the source benchmark as a hypothesis, not as universal support evidence:
- If $N \le 200$, benchmark Jacobi against divide-and-conquer on the target
  DTK version and dtype.
- If $N > 200$, benchmark divide-and-conquer against Jacobi on the target
  workload.

### 3. Handle Memory Layout & Strides

PyTorch tensors are row-major; LAPACK and rocSOLVER expect column-major.
For symmetric matrices, row-major lower-triangle storage corresponds to
column-major upper-triangle storage. Invert uplo before passing it to
rocSOLVER, and transpose the in-place eigenvector output back to the PyTorch
layout when required:

    V = A_work.transpose(-1, -2).contiguous()

### 4. Stream & Handle Management Invariants

- Maintain a thread-local rocblas_handle cache keyed by device ID.
- Query the current PyTorch stream from the input device.
- Bind the current stream with rocblas_set_stream before enqueueing kernels.
- Set the active device before handle or memory operations for multi-DCU safety.

### 5. Apply the Three-Layer Validation Gate

1. **API Layer**: check rocblas_status.
2. **Solver Layer**: check info and Jacobi sweep/convergence outputs.
3. **Numerical Layer**:
   - Compare eigenvalues with PyTorch or CPU LAPACK.
   - Check orthonormality.
   - Check normalized reconstruction residual.

Do not claim production support from the source benchmark alone. Record the
actual device, DTK version, dtype, matrix sizes, batch sizes, tolerances, and
the exact evidence path.

## Integration in BFGS & Spectral Optimizers

For a Hessian decomposition
$$H_k = V \Lambda V^T$$
with regularized inverse curvature, a batched spectral step has the form:

    w, V = eigh_batched(H_batch, UPLO="L")
    proj_g = torch.bmm(V.transpose(-1, -2), grad_batch)
    w_clamped = torch.clamp(w, min=min_curv).unsqueeze(-1)
    step = -torch.bmm(V, proj_g / w_clamped)

The source package torch_rocsolver_eigh is an external reference and is not
part of this repository's framework API. Integrating an eigensolver here still
requires the repository's operation contract, registry metadata, executor
binding, framework wiring, CPU oracle, and focused tests.

## Diagnostics & Troubleshooting

| Symptom | Root Cause to Check | Fix to Benchmark |
| :--- | :--- | :--- |
| Batch time scales linearly for small matrices | Possible syevd host fallback | Compare syevj and syevd with tracing enabled |
| Eigenvector reconstruction error is near 1 | Layout or transpose mismatch | Invert uplo and restore the eigenvector layout |
| rocblas invalid value | Stride or LDA misconfigured | Check lda, strideA, and strideW |
| Multi-card hang | Handle stream or device conflict | Use per-device handles and bind the current stream |
| Unclear solver path | Missing low-level trace | Use ROCSOLVER_LAYER=1 or ROCSOLVER_LAYER=2 |

## References

- [Hybrid Dispatch & Pitfalls](references/hybrid-dispatch-and-pitfalls.md):
  source-specific analysis of the small-matrix fallback and benchmark evidence.
- [Memory & Stream Contracts](references/memory-and-stream-contracts.md):
  column-major vs row-major derivations and handle lifecycle rules.
- [BFGS Spectral Optimization](references/bfgs-spectral-optimization.md):
  mathematical formulations for crystal relaxation and batch Hessian updates.
