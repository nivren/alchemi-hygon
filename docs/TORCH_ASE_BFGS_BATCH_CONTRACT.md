# ASE-compatible BFGS M4 Batch contract

M4 is an authorized local extension, not a port of an upstream BFGS. It retains
the ASE 3.29.0 formula and the M1–M3 single-system interfaces and snapshots.

## Numerical API

`bfgs_ase_batch_step(positions, forces, states, backend=..., selection=...)`
accepts equally long, nonempty sequences of independent contiguous FP64 tensors
and `ASEBFGSState`. Each coordinate/force tensor has `[d_i/3,3]` shape. All
inputs have one device and coordinate mode, unique system IDs, and no shared
storage. Cartesian and ASE UnitCellFilter coordinates use the same core.
State remains independently owned; no global padded Hessian is allocated.

The executor groups by scalar DOF and stacks each group for vectorized Hessian
updates, batched symmetric decomposition, direction calculation and per-system
maxstep. Only the ASE displacement guard skips an update. No damping, spectrum
clamp, curvature criterion or precision/device conversion is introduced.
All groups must produce valid candidates before any input/state is committed.
Invalid input, exact-zero denominators/eigenvalues or solver failure raise.
Non-finite directions and step lengths also raise before any group commits.
A finite direction whose norm overflows must not silently become a zero step;
this failure boundary matches the single-system executor.
Finite but invalid eigensystems also fail before commit: relative Frobenius
equation residual and identity-normalized orthogonality must each be <=1e-10.
This checks solver validity, without modifying eigenvalues or vectors.
Forces are read-only; positions/history/Hessian/count mutate on success. Returned
diagnostics own independent storage. Empty sequences are rejected.

Torch is the reference executor. Explicit HIP reuses the validated native
batched rocSOLVER eigensolver. On the admitted M4 build all widths use checked Jacobi after
a captured CSP Hessian exposed a fault/incorrect spectrum in batched DC. No
failure-triggered solver switching is used. Current Torch stream applies throughout; no CPU
numerical fallback. Eager inference only, no autograd/compile contract. ASE
units, row-cell convention, PBC and neighbor semantics remain M1–M3's; this
operation consumes forces and does not construct neighbors.

## Framework lifecycle

The existing `ASECompatibleBFGS` and `ASECompatibleBFGSVariableCell` accept
heterogeneous Batch. Multiple systems require unique integer `system_id`.
State, identity signature, convergence freeze and optional Filter/reference cell
are keyed by ID, never slot. Reorder retains history; removal releases state;
arrival initializes alpha*I and empty history. Reuse of an active ID with changed
identity/DOF/cell (fixed-cell) is rejected. Variable-cell Filter options apply to
each system; default cell_factor is its atom count. No slot-indexed parameters.

Model evaluation remains the existing shared Batch call. Standalone steps prime
once and freeze converged systems. Masked FusedStage entrances consume the
already primed/shared forces and never evaluate the model themselves. Only
eligible, unconverged systems mutate BFGS history. Native sampler budget/refill,
Hooks, status migration and sinks remain responsible for workflow scheduling.
No change to FIRE/FIRE2 or generic FusedStage is planned.

A separate versioned Batch snapshot stores per-ID single-system continuation
snapshots and frozen flags. Restore validates all entries atomically, supports
explicit whole-state device movement, and checks paired physical geometry on
next entrance. Saving models, Batch/status/counters, dataset/sampler cursor and
pipeline scheduling is the caller's responsibility. M2/M3 single-system formats
continue to load and save when using their standalone path.
Promoting a loaded single-system snapshot into Batch retains its required paired
ID and geometry check; additional systems initialize fresh histories.

## Acceptance

CPU ASE stepwise oracle and independent-run parity for heterogeneous fixed/variable
cells; equal-DOF batched decomposition evidence; failure atomicity; convergence
freezing; reordering, exit/refill and restart; shared model call counts; existing
CPU gate. Bounded available-card HCU validation includes native batching/current
stream/device residency. Real 46/92-atom molecular-crystal MPA-0 FIRE2→BFGS
workflow records convergence, model calls, inputs, numerical differences and
timings. Short budgeted lifecycle runs are distinguished from successful CSP
relaxation. Profiling determines a later M5 proposal, not automatic optimization.
