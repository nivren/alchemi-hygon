# ADR 0005: opt-in no-PBC Torch reference cell-list

- Status: proposed implementation slice
- Date: 2026-09-08
- Scope: `nvalchemiops` neighbor dispatcher and framework `compute_neighbors`

## Decision

Add `torch_reference_cell_list` as an explicit, capability-registered backend for
no-PBC neighbor construction. Keep `torch_reference` as the semantic oracle and
keep `None`/`warp` as the unchanged upstream default. `auto` does not select the
cell-list backend in this slice; choosing it requires a later benchmark-backed
priority decision.

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

The backend can be compared against the existing dense `torch_reference` on the
same inputs before it is considered for `auto` or production use. Framework
one-shot `compute_neighbors` accepts the new backend; dynamic Hook support uses
the existing reference staging/grow contract but remains outside this first
performance claim. The default path and all periodic behavior are unchanged.
