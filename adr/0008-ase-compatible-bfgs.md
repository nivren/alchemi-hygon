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
without an explicit adapter and separate validation. The original planning decision
itself created no HCU capability, HIP implementation or performance claim. Actual M1
implementation evidence is recorded separately below.

Evidence and task ownership are recorded in `docs/PARALLEL_DEVELOPMENT_PLAN.md`,
`docs/FEATURE_COMPATIBILITY.yaml`, `docs/STATUS.md` and `docs/PROJECT_HANDOFF.md`.

## M1 implementation decision, 2026-10-09

BFGS is a **local new function absent from the reviewed NVIDIA upstream**, not an
upstream operation port or bugfix. Architecture references frozen Framework
0.3.0-rc/Ops 0.5.0-rc L-BFGS APIs; the product lock remains 0.2.0/0.4.1. The independently
written mathematics uses frozen ASE 3.29.0 sources as an oracle, without an ASE runtime
dependency or copying LGPL source into the Apache modules.

M1 provides `nvalchemiops.bfgs_ase` State/prepare/step and device-local Torch executor,
registered as `torch_reference.bfgs-ase-v1`. The new API defaults to explicit
`torch_reference`; absent legacy BFGS and unadmitted auto fail. Cached selections are
validated, with no generic executor edits. Exact zero denominators/eigenvalues and
non-finite candidates fail before mutation; no near-zero threshold or algorithmic
safeguard is introduced. Restart is a versioned local tensor snapshot, not ASE JSON.

The user approved M1–M4 (M1 Ops, M2 Framework, M3 ASE UnitCellFilter, M4 heterogeneous
Batch/CSP). Filter defaults follow ASE, with full parameter verification in M3.
M5 is optional, conditional on measured end-to-end bottlenecks. Each next stage requires
user review and confirmation. M1 CPU/HCU evidence does not deliver M2–M4.

See [contract](../docs/TORCH_ASE_BFGS_CONTRACT.md),
[approved roadmap](../docs/ASE_BFGS_DEVELOPMENT_PLAN.md) and
[M1 evidence](../reports/g2-ase-bfgs-m1.md).

## M2 implementation and eigensolver finding (2026-10-09)

The approved M2 slice adds public `ASECompatibleBFGS` as a BaseDynamics subclass,
reusing the model, update hooks and run loop. Its small local step boundary primes
initial forces and checks strict ASE stopping before updates. Dense Ops state is
kept separately from the base system-only Batch state; inactive single-system
callbacks skip all math, preserving Hessian, histories and update metadata.
Scheduling visits and numerical updates are distinct counters. Restart validates
the paired geometry and default/custom stopping mode; custom hooks are recreated
by the caller. No shared Framework engine/state-manager changes were made.

Actual HCU tracing found host `LAPACKE_dstedc` and tridiagonal D2H/eigenvector H2D
inside Torch for n=138 and n=199. No monitored LAPACK calls appeared at n=200/201,
so the external prototype's `n <= 200` policy is not a validated product rule.
The isolated ctypes rocSOLVER candidate preserves inputs and checks API status,
solver info and eigensystem residual/orthogonality. It is not registered as a
product backend. Small B=1 Jacobi is slower; B=16 has measured solver-only gains.

M2 validates a narrow numerical/reference integration, not entirely HCU library
execution or production CSP performance. The next proposed E1 stage adapts an
explicit HCU eigensolver and validates host-work exclusion, streams and ASE
trajectories while retaining the reference and unchanged BFGS formulas. E1 requires
user confirmation, followed by independently confirmed M3/M4.
See [M2 evidence](../reports/g2-ase-bfgs-m2.md).


## E1 native solver adaptation (2026-10-09)

The user approved E1 after M2 demonstrated host mathematics. This is a device-path
correctness adaptation, separate from optional M5 end-to-end CSP optimization.
An independently written BFGS-private extension uses rocSOLVER, not the external
external reference project's code. The generic registry/executor and Framework
engine remain unchanged; an explicit HIP executor delegates to the same numerical
BFGS body with a solver callable. The shared HIP loader only gains optional linker
flags, whose empty default preserves existing native neighbor builds.

The admitted HCU Torch/HIP/rocSOLVER build is guarded explicitly. Current-library
traces justify Jacobi for n<200 and DC at/above200; unknown builds must requalify.
Device guards, current streams, per-thread/device RAII handles, independent row
storage, strict UPLO and GPU solver-info checks fix the audited external extension
risks. Forward-only custom-op schema/fake checks do not claim compile or Autograd.

Product eigh and whole BFGS traces show no monitored LAPACK or matrix roundtrip at
validated sizes; diagnostic scalars and library index metadata remain host-visible.
Small B=1 Jacobi is slower, so explicit HIP does not imply a speedup. Equal-size
solver batches have measured gains, but the optimizer itself remains single-system
until M4. See [E1 evidence](../reports/g2-ase-bfgs-e1.md) and
[solver contract](../docs/HIP_BFGS_EIGH_CONTRACT.md). M3 is proposed next, awaiting
confirmation; M3/M4 and complete feature verification remain outstanding.

## M3 full UnitCellFilter chart (2026-10-09)

The user confirmed M3 and asked explicitly about upstream L-BFGS's different
variable-cell representation. The frozen Ops 0.5.0-rc source at 5fffc072 uses
six lower-triangular components, packed as N+2 vec3 rows after cell alignment.
ASE UnitCellFilter uses the full deformation gradient relative to the reference
cell, packed as N+3 rows. Nine matrix entries do not mean nine independent
physical strains. These charts have different force mappings and maximum-step
norms; zero padding cannot establish ASE compatibility. The local frozen source
hash verification is retained in the M3 artifacts, without changing product locks.

M3 adds independently written device-local `ASEUnitCellFilter` Ops and public
`ASECompatibleBFGSVariableCell`, preserving all ASE Filter parameters, default
scales and projection order. It reuses the shared BFGS math and existing
BaseDynamics/Hooks, without native cell alignment, a Filter plugin framework,
or a new executor. Default stopping is the strict norm of generalized force
rows; Batch energy remains physical and enthalpy is available separately.

Coordinate-mode metadata separates Cartesian schema 1 snapshots (unchanged)
from UnitCell schema 2 snapshots. The variable-cell Framework restart additionally
saves the Filter reference/parameters and paired physical geometry. Scratch
history protects the complete geometry and Hessian when an unpacked cell is
singular/nonfinite; this adds O(d²) device-local copies, recorded as a performance
cost rather than an algorithm change. The shared numerical function's AST and
native eigensolver files remain equal to E1. Constant-volume projection follows
ASE and does not enforce exact per-step volume. Constraints/other Filters are
outside this slice; heterogeneous Batch/refill/CSP remain the mandatory M4 scope.

Contract: [UnitCellFilter](../docs/TORCH_ASE_UNITCELL_FILTER_CONTRACT.md).
Evidence and current limits: [M3 report](../reports/g2-ase-bfgs-m3.md).

## M4 approved extension and solver correction (2026-10-09)

The user confirmed M4 with "continue". Keep independent per-system State and
Filter, keyed by stable ID in a private optimizer adapter. Group equal scalar
DOFs for vectorized Hessian updates and batched eigendecomposition; do not pad
or truncate spectra. Reuse the existing shared model, Hooks, FusedStage, sampler
and sinks. Reorder retains histories, departures release them, arrivals start
fresh, and convergence freezes each system. Validate all numerical groups and
unpacked cells before committing. Batch snapshots validate all ID/geometry pairs;
promoting a legacy single-system restart also retains its paired geometry guard.

A real CSP 285-DOF near-alpha degenerate Hessian exposed VMFault in the admitted
rocSOLVER strided-batched DC path; isolated replay also returned wrong finite
eigenvalues with info=0. E1's size threshold is superseded: the private HIP solver
now uses checked Jacobi directly at all widths. This is a fixed implementation
choice, not failure-triggered fallback or a change to the ASE spectrum/formula.
Add device-local residual/orthogonality validity checks before State mutation.
Preserve the captured input, archived DC sources and failed runs. Other solver
interfaces/builds are not presumed faulty; HCU Torch solved that captured matrix.
This is a local new feature/private library-call correction, not an NVIDIA UBF.

M1–M4 passed the scoped CPU/HCU acceptance, including real 46/92-atom MPA-0
FIRE2-to-BFGS convergence and four inflight relaxations. All work remains
uncommitted/unintegrated; no product lock upgrade or generic engine/FIRE change.
Only optional M5 performance work remains, subject to a new user confirmation.
Dense State/scratch cost and measured eigensolver bottlenecks motivate a bounded
warm benchmark/candidate decision, not speculative kernels or automatic DC reuse.
No NVIDIA hardware, other HCU build, large CSP success rate, distributed/compile
or general pipeline checkpoint claim follows from these tests.

Contract: [Batch](../docs/TORCH_ASE_BFGS_BATCH_CONTRACT.md).

## M5-4 implementation decision (2026-10-10)

The approved integration registers an explicit
`hip.bfgs-ase-jacobi-streams-v1` executor with a separate `jacobi_streams`
strategy, neither default nor auto-enabled. `hip` retains rocSOLVER Jacobi.
The local original dynamic HIP Jacobi solves one width as a batch; multiple
widths share two persistent host workers/streams on one HCU. Worker count is
not a Batch capacity limit. No empirical dimension/Batch whitelist is added.
The shared ASE mathematical expressions and state/restart/Filter contracts
remain; all groups validate before committing any state. Event dependencies,
allocator stream recording and draining accepted work on failures protect
lifetime. Generic framework dynamics and registry dispatch are unchanged.

A measured zero-sweep centering cancellation erased valid 1e-18 eigenvalues.
The final native output path sorts the original diagonal on device for those
members, preserving tiny values, signed zero and stable eigenvector order.
Iterative rotation kernels remain unchanged; frozen probes are retained rather
than altered to hide the defect. This is a solver precision correction, not
curvature damping, spectrum clamping or a new BFGS algorithm.

The backend is opt-in, input-dependent and qualified only for the recorded
FP64 eager HCU build. Fake metadata is not compile or autograd acceptance.
See [contract](../docs/HIP_BFGS_JACOBI_STREAMS_CONTRACT.md) and
[validation report](../reports/g2-ase-bfgs-m5-4.md). No shared-branch integration,
publishing or default selection is authorized by this milestone.
Evidence and limits: [M4 report](../reports/g2-ase-bfgs-m4.md).

## Local delivery record (2026-10-11)

The user approved four local DCO-signed commits for Ops, Framework, validation
records and delivery documentation. Product/test/probe contents match the final
reviewed snapshot; the documentation commit records the local delivery state.
No merge into `develop`, push, publishing, default change or wider hardware
qualification follows from this approval. See the
[delivery record](../docs/ASE_BFGS_DELIVERY_PLAN.md) and the
[final review](../reports/g2-ase-bfgs-delivery-review.md).
