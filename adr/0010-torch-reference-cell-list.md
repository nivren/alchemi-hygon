# ADR 0010: opt-in no-PBC Torch reference cell-list

- Status: superseded in part by ADR 0006
- Date: 2026-09-08
- Scope: `nvalchemiops` neighbor dispatcher and framework `compute_neighbors`

## Decision

At the time of this decision, `torch_reference_cell_list` was introduced as an
explicit, capability-registered backend for no-PBC neighbor construction. Keep
`torch_reference` as the semantic oracle and keep `None`/`warp` as the unchanged
upstream default. `auto` does not select the cell-list backend in this slice;
choosing it requires a later benchmark-backed priority decision.

## Contract

- Input positions are `float32` or `float64`, shape `(N, 3)`, with contiguous
  `batch_ptr`/`batch_idx` system boundaries. Empty systems and an empty batch are
  valid.
- The cell-list path supports no-PBC `full` and `half` lists, MATRIX and COO
  output, optional distances/vectors, deterministic row-major `(source,target)`
  ordering, strict `distance < cutoff`, and explicit capacity overflow errors.
- Self pairs are excluded. Active overlapping atoms raise an error. Pairs never
  cross a batch system. Topology is non-differentiable; returned continuous
  distances/vectors retain the ordinary Torch position dependency.
- A uniform Cartesian cell edge equal to the cutoff is used. Candidate pairs are
  generated from the 27 neighboring cells and then filtered by the exact cutoff.
  Host control is limited to system/cell metadata; positions and candidate
  tensors remain on the target device.
- Periodic cells, periodic half lists, target rows, pair callbacks, scratch
  buffers and dynamic rebuild arguments fail explicitly. They are separate
  capability slices and are not silently routed to the dense reference path.

## Consequences

The implementation can be compared against the existing dense
`torch_reference` on the same inputs before it is considered for `auto` or
production use. Framework one-shot `compute_neighbors` accepts the strategy;
dynamic Hook support uses the existing reference staging/grow contract but
remains outside this first performance claim. The default path and all periodic
behavior are unchanged.

## Supersession

ADR 0006 replaces the public backend-name decision: the implementation is now
selected as `backend="torch_reference", method="cell_list"`, not by the
unpublished `torch_reference_cell_list` global request. This ADR remains the
algorithm and narrow-contract record for the implementation.

## 2026-09-19 periodic core extension

`TORCH-NEIGHBOR-PBC-CELL` extends the same explicit strategy to periodic and
mixed-PBC orthogonal/triclinic cells, periodic half lists, layered build/query,
preallocated scratch, batched selective rebuild, and differentiable continuous
vector/distance outputs.  Search radii are derived conservatively from reciprocal
cell-vector norms, so correctness does not depend on a fixed 27-cell stencil.

This extension does not change `backend=None`, Warp, or `auto`.  Target rows,
pair callbacks/outputs, pair-centric/sorted query, compile/opcheck, and optimized
HIP/Triton implementations remain separate capabilities.  The measured Torch
implementation is a correctness oracle rather than a production choice: on the
recorded BW200 workloads it is slower than dense, so no automatic selection is
authorized.  Full evidence and exact boundaries are recorded in
`reports/g2-torch-reference-pbc-cell-list-core.md`.
