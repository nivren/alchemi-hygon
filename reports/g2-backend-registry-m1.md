# G2 M1 implementation registry and neighbor strategy

## Scope

M1 separates backend request, implementation family, stable implementation ID
and neighbor strategy. It registers legacy Warp, dense Torch reference and
no-PBC cell-list metadata without introducing a platform profile or planner.
Cell-list is selected with ``backend="torch_reference", method="cell_list"``;
the unpublished global ``torch_reference_cell_list`` request now fails.

## Contract and results

- ``backend=None`` remains the framework-owned legacy Warp selection with
  implementation ID ``warp.legacy-upstream-v1``.
- ``auto`` continues to select only the dense Torch reference default strategy;
  requesting a strategy through ``auto`` fails until M2 profile support.
- Framework passes one resolved selection to the Torch dispatcher. Dispatcher
  chooses dense or cell-list by implementation ID, not by a family string.
- CPU: ops registry/reference/cell-list tests passed ``25`` tests; framework
  neighbor/Hook tests passed ``22`` tests. Each suite emitted one expected
  ``auto`` selection warning.
- BW200/UBB BW1000 (gfx936), source DTK 26.04, project ``.venv``,
  ``HIP_VISIBLE_DEVICES=0``: ops registry/reference/cell-list tests passed
  ``25`` tests in 19.39 s; framework one-shot and Hook cell-list strategy tests
  passed ``2`` tests in 16.84 s. Both commands exited 0.

## Reproduction

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/ops OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_backend_registry.py \
  packages/ops/test/torch/test_torch_reference_backend.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py

PYTHONPATH=packages/framework:packages/ops OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py
```

For the HCU smoke, add ``HIP_VISIBLE_DEVICES=0``. The full framework suite was
started on HCU but its compiled-entrypoint output did not produce a recoverable
completion record in this execution carrier, so it is not claimed as passed.
The two non-compiled M1 strategy tests above are the HCU framework evidence.

## Limits

This is not a BackendProfile, PipelinePlanner or Frozen BackendPlan. It does
not approve ``auto`` to choose cell-list, add periodic cell-list, establish a
performance result, or change Warp defaults.
