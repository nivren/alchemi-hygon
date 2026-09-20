# ADR 0009: HIP cell-list build uses an explicit staged pipeline with selected fusion

- Status: accepted
- Date: 2026-09-19
- Scope: `TORCH-NEIGHBOR-PBC-CELL` stage-two HIP build/binning path; not query, dispatcher, or default selection

## Context

The periodic Torch cell-list is the correctness oracle, but its build path currently
materializes a linear key, calls Torch count/scan, performs a stable sort, then queries
the resulting CSR storage. The stage-one HCU trace showed cost distributed across many
small indexing, sort and scan/reduce launches; isolated microkernels are therefore an
insufficient design target.

The locked upstream Warp build is an algorithmic reference. It runs `construct_bin_size`,
then a per-atom `count_atoms` kernel that computes PBC/cell coordinates and atomically
increments cell counts. It performs an exclusive scan into cell starts, clears counts,
and runs a second per-atom `bin_atoms` kernel that uses counts as atomic insertion
cursors and restores final counts while writing `cell_atom_list`. This does not prove
that its Warp/CUDA implementation is optimal on DCU.

Existing stage-two evidence is deliberately narrower: native cell-key and atomic-count
custom ops have output parity; the count-only API has a 1.30x microbenchmark factor on
one BW200 configuration. Neither candidate is connected to the reference build path.

## Decision

### Pipeline and semantic boundaries

The future HIP build is one explicit pipeline with device-resident buffers:

```text
grid metadata
  -> fused geometry/PBC/key-count
  -> global exclusive scan
  -> fill CSR atom list
  -> query/materialize neighbors
```

- Grid sizing, capacity decisions and public error behavior remain governed by the Torch
  reference contract until their own device path is specified and verified.
- The fused geometry/count phase may write public `atom_periodic_shifts` and
  `atom_to_cell_mapping` while atomically producing counts. A linear key buffer is kept
  only when a separately selected consumer needs it; it is not a required permanent
  intermediate of the optimized pipeline.
- Scan is a separate phase. A general exclusive scan cannot be fused with atom counting
  without a global completion boundary. Its future custom op reads active `int32` counts,
  writes equally shaped `int32` starts plus the supplied global atom offset, and must not
  modify counts. Workspace query/allocation is explicit and amortized outside steady
  timing.
- Fill is a separate per-atom phase after scan. It may follow the upstream reuse pattern:
  zero the private build-time count buffer, atomic-add it as a per-cell insertion cursor,
  and leave the restored final counts after all inserts. No reader may observe this
  transient cursor state. A distinct cursor scratch buffer remains an alternative when a
  later ordering or concurrency contract requires counts to stay readable during fill.
- The current reference's stable `argsort` is its oracle implementation, not permission
  to relax public ordering. An optimized fill may have unspecified internal per-cell atom
  order only after the public neighbor MATRIX/COO ordering, pair/shift set, capacity and
  gradient contracts are reproduced by an explicit canonicalization/test boundary.

### DTK implementation choice

`/opt/dtk-26.04/include` contains `rocprim/device/device_scan.hpp` and
`hipcub/device/device_scan.hpp`, including `rocprim::exclusive_scan`; the active
`/opt/dtk-26.04/bin/hipcc` reports dcc 25.10.0 / clang 17. Therefore rocPRIM is the
first scan candidate, using its two-call temporary-storage query followed by an
explicit-stream execution. Header/tool discovery alone was not compile or HCU execution
evidence; the standalone boundary subsequently JIT compiled and ran on BW200/gfx936 with
output parity, empty input, strided counts and a non-default stream. This remains only
the scan phase evidence, not complete build-pipeline or performance evidence.

The existing HIP key/count kernels remain independent correctness candidates. They must
not be silently chained or selected by `auto`. A fused key-count implementation is a
new operation with its own output, overflow, stream and benchmark evidence, rather than
an undocumented edit to either candidate.

### Performance decisions

- Compare device-event steady-state latency under identical dtype, keys/positions, cell
  occupancy, device and buffer ownership. Keep raw samples; separate JIT, isolated phase,
  complete build and end-to-end neighbor timing.
- Measure both uniform occupancy and concentrated-cell contention. The current 1.30x
  count-only result justifies retaining the candidate, not backend registration.
- Profile only a bounded, focused candidate after correctness and comparable baseline
  exist. Do not transfer Warp block sizes, assume a HIP primitive is faster, or tune a
  phase solely from source appearance.
- `auto` remains unchanged until a representative complete neighbor workload reaches the
  existing 2x neighbor or 20% target end-to-end gate with the full numerical contract.

## Consequences and sequence

1. Implement a standalone rocPRIM exclusive-scan boundary with fixed active slice,
   global offset, workspace contract, CPU rejection and HCU parity. It does not yet
   connect to full build.
2. Define and validate a fused geometry/PBC/count candidate against the existing key and
   count oracles. Benchmark it against the composition of their explicit APIs.
3. Implement fill with explicit cursor lifetime, then compare its CSR content and public
   neighbor outputs against the reference for PBC/no-PBC, half/full, triclinic and Batch.
4. Only then benchmark build and query together, and separately decide whether a native
   query path or a Triton materialization path is justified.

As of 2026-09-19, steps 1 and 2 have separate CPU/HCU correctness evidence. Step 3 has a
CSR fill ABI/Torch oracle, a native atomic candidate with per-cell-set (not stable-order) HCU
evidence, and an isolated three-phase Batch CSR composition with matching metadata/set evidence.
Its HIP build plus the existing Torch direct query now has public-neighbor-order evidence;
pair-centric and other deferred query surfaces remain unverified. Step 2 uses
a Batch-aware global-cell-offset ABI and one native kernel for geometry/PBC/key/count.
Its same-contract two-system FP32 device-event comparison versus the independent native
key-plus-count composition measured a 1.481x/1.483x factor for uniform/single-cell
occupancy; this retains the candidate but is not complete build or neighbor evidence. No
step has connected these candidates to the runtime build path.

This ADR creates no new public backend, no AOT artifact requirement, and no claim that
the upstream's atom-centric/pair-centric query strategies or full feature surface are
implemented on DCU.

The first complete build-plus-direct-query device comparison (fixed FP32 2x2048 Batch,
uniform/clustered) found native build slower by about 18.4%, query approximately tied, and
full native slower by 2.9%--3.4%. This is a narrow pre-warmed device-event result, not an
`auto` gate; it directs the next investigation toward query/materialization.
