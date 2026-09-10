# ADR 0008: ASE-compatible BFGS is a distinct optimizer contract

- Status: accepted
- Date: 2026-09-10
- Scope: BFGS structure-relaxation API, numerical oracle and HCU optimization boundary

## Context

The framework needs a no-line-search BFGS path comparable to a prior molecular-crystal
relaxation workflow.  The project environment contains `ase==3.29.0`.  Its public
`ase.optimize.BFGS` delegates the numerical step to `ase._4.optimize.bfgs.BFGSMethod`:
it maintains a dense Hessian approximation, updates it with the BFGS Hessian formula,
uses `eigh(H)`, replaces eigenvalues by their absolute values to compute the descent
direction, and applies a global maximum-step scale.  This is not interchangeable with
a conventional inverse-Hessian BFGS, L-BFGS, a line-search optimizer, or an optimizer
that silently skips/damps curvature updates.

ASE variable-cell compatibility is also materially different from the existing native
cell-filter direction: `UnitCellFilter` represents atomic coordinates plus a 3x3
deformation gradient (`3N+9` degrees of freedom), including `cell_factor`, mask,
hydrostatic/constant-volume and pressure semantics.  The current framework-native
cell filter is an aligned upper-triangular `3N+6` representation.

## Decision

- Register `TORCH-BFGS-ASE-COMPAT` as a P1 planned operation.  Its first release slice
  is one fixed-cell system; heterogeneous Batch, inflight and DomainParallel are
  explicitly unsupported rather than padded or made to share Hessian state.
- The fixed-cell Torch reference must use ASE 3.29 as a CPU FP64 step-level oracle for
  Hessian state, direction, maximum-step clipping and restart.  In the compatibility
  mode, the ASE Hessian/eigenspectrum algorithm is the contract.
- Do not introduce a generic public eigensolver operation or change the generic executor
  to serve BFGS.  The BFGS implementation owns its internal linear-algebra selection.
- `TORCH-BFGS-ASE-UNITCELL` is a later dependent milestone.  It must implement the
  `UnitCellFilter` `3N+9` contract and must not describe the native `3N+6` cell-filter
  route as ASE-compatible.  The original workflow's chosen ASE filter must be recorded
  before claiming full variable-cell trajectory compatibility.
- `torch.linalg.eigh` is the initial HCU correctness and benchmark path.  A HIP
  eigensolver is conditional on warmed FP64 measurements at target dimensions proving
  `eigh` is a stable end-to-end bottleneck.  Any optimized path remains subject to the
  same ASE CPU FP64 step oracle; no silent algorithm substitution is permitted.

## Consequences

The first implementation has a deliberately narrow, reviewable state boundary and a
clear numerical comparator.  It may be slower than an L-BFGS or a native safeguarded
BFGS, but those are distinct future modes with their own contracts and evidence.
Variable-cell ASE compatibility cannot reuse the current FIRE2 native cell-filter API
without an explicit adapter and separate validation.  The decision creates no current
HCU capability, HIP implementation or performance claim.

Evidence and task ownership are recorded in `docs/PARALLEL_DEVELOPMENT_PLAN.md`,
`docs/FEATURE_COMPATIBILITY.yaml`, `docs/STATUS.md` and `docs/PROJECT_HANDOFF.md`.
