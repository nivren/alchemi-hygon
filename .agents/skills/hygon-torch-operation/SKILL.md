---
name: hygon-torch-operation
description: Implement or review a new Torch reference operation in alchemi-hygon, including registry metadata, executor binding, framework wiring, and focused CPU/HCU validation. Do not use for generic Python changes, documentation-only edits, or already-supported behavior.
---

# Hygon Torch operation

Use this skill only for one new operation, or one auditable wiring step for a
new operation, in the Torch/reference backend path.

## Before editing

- Read the relevant sections of `docs/ADD_TORCH_OPERATION.md`,
  `docs/DEVELOPMENT_GUIDE.md`, and `docs/TEAM_DEVELOPMENT_BASELINE.md`.
  Inspect the locked source under `packages/` and its tests; keep `external/`
  read-only.
- Write or confirm the operation contract before implementation: shape, dtype,
  layout, units, PBC and neighbor convention, empty/capacity/error behavior,
  mutation/aliasing, stream/determinism, and gradient level.

## Implementation invariants

- Make the Torch path the semantic baseline and retain a CPU oracle and
  focused regression tests. Do not hide missing gradients with `detach`, or
  silently change device, dtype, precision, neighbors, or interactions.
- Register a unique implementation entry in the ops catalog. Keep catalog
  metadata declarative: it must not import Torch, Warp, HIP, or execute code.
- Resolve `BackendSelection` once in framework code and pass it through the
  fixed dispatcher ABI to generic `execute_selected`/executor bindings. Use
  lazy executor imports; do not add an implementation-ID dispatch table.
- Preserve `backend=None` legacy Warp behavior. Unknown or unsupported explicit
  requests must fail clearly; never add a silent fallback.

## Validation and handoff

- Run the affected CPU tests and include the operation in
  `scripts/check_cpu_reference.sh` when appropriate.
- Run a bounded HCU smoke only when the change touches an HCU path and an
  allocated device is available. If no device is available, record HCU as
  pending and continue without waiting for one.
- Update `docs/FEATURE_COMPATIBILITY.yaml`, `docs/STATUS.md`, or a report only
  when the change adds a fact, support state, or evidence. Keep planned,
  implemented, verified, blocked, and deferred distinct.
- Handoff must separate implementation, tests, numerical evidence, and
  unverified scope. The expected result is one operation or one auditable
  wiring step; split planner/global-policy, cross-rank, or production-
  performance work into a separately scoped design task.
