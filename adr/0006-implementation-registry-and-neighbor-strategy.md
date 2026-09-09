# ADR 0006: implementation registry and neighbor strategy

- Status: accepted
- Date: 2026-09-09
- Scope: M1 backend registry semantics and Torch neighbor dispatch

## Context

The prior capability table used one ``BackendName`` literal for policy tokens,
backend families, the temporary cell-list implementation and legacy Warp. That
made a neighbor algorithm look like a global execution backend and required
callers to repeat an incomplete backend enum.

## Decision

- Replace the fixed public backend literal with open ``BackendRequest = str |
  None`` and resolve it only through ``ImplementationRegistry``.
- Register stable implementation IDs for legacy Warp, Torch dense neighbors,
  Torch no-PBC cell-list and the existing Torch reference operation slices.
  Registry metadata records family, operation, strategy, lazy executor path,
  contract width and evidence; it does not profile or import executors.
- Preserve ``backend=None`` and ``backend="warp"`` as framework-owned legacy
  Warp selections. ``auto`` may select only a default strategy and never
  selects an operation strategy before a BackendProfile exists.
- Expose cell-list as
  ``backend="torch_reference", method="cell_list"`` on
  ``compute_neighbors`` and ``NeighborListHook``. The unpublished
  ``torch_reference_cell_list`` request is removed and fails explicitly.
- Pass the one framework-resolved ``BackendSelection`` to operation dispatchers;
  neighbor, LJ, VV/FIRE/FIRE2, periodic, kinetics and observer helpers dispatch
  by implementation ID rather than by family or by resolving the request again.
- Keep FIRE and FIRE2 as separate operation contracts and stable IDs. Variable-cell
  FIRE/FIRE2 request the ``variable_cell`` feature during framework initialization;
  the current Torch reference implementations do not claim that capability.

## Consequences

M1 does not create PlatformFingerprint, BackendProfile, Frozen BackendPlan,
runtime fallback policy or whole-pipeline planning. No default path changes:
``None`` remains legacy Warp and ``auto`` still resolves only the dense Torch
reference capability currently registered. ADR 0005 remains the cell-list
algorithm contract, but its public backend-name decision is superseded here.

CPU and BW200/gfx936 HCU contract evidence is recorded in
`reports/g2-backend-registry-m1.md`; the operation selection propagation follow-up
is recorded in `reports/g2-backend-registry-m1-dispatch-propagation.md`. The cell-list
path remains a no-PBC, reference-only strategy without real-structure performance
evidence, and the new propagation evidence is CPU-only.
