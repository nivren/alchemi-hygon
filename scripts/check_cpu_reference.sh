#!/usr/bin/env bash
# Run the stable Torch-reference development baseline on CPU.
#
# The framework and ops upstream test suites both use a top-level ``test``
# package, so this script deliberately starts them in separate pytest
# processes. It is the required local gate for changes proposed to the team
# baseline; it is not a replacement for an HCU evidence run.

set -euo pipefail

_baseline_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$_baseline_root"

# shellcheck source=/dev/null
source scripts/activate_hygon_env.sh project

_baseline_python="$_baseline_root/.venv/bin/python"
export PYTHONPATH="$_baseline_root/packages/framework:$_baseline_root/packages/ops${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NVALCHEMI_TEST_BACKEND=torch_reference

"$_baseline_python" -m pytest -q \
  packages/ops/test/torch/test_backend_registry.py \
  packages/ops/test/torch/test_torch_reference_backend.py \
  packages/ops/test/torch/test_torch_reference_cell_list.py

"$_baseline_python" -m pytest -q \
  packages/framework/test/models/test_lj_torch_reference.py::test_reference_model_builds_neighbor_hook_without_eager_dynamics_import

"$_baseline_python" -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_import_boundary.py \
  packages/framework/test/compatibility/test_executor_binding.py \
  packages/framework/test/compatibility/test_backend_selection_propagation.py \
  packages/framework/test/compatibility/test_dynamics_reference_velocity_verlet.py \
  packages/framework/test/compatibility/test_dynamics_reference_fire.py \
  packages/framework/test/compatibility/test_public_dynamics_reference.py \
  packages/framework/test/dynamics/test_hook_utils.py \
  packages/framework/test/dynamics/test_periodic_hook.py \
  packages/framework/test/dynamics/test_observer_hooks.py \
  packages/framework/test/models/test_lj_torch_reference.py \
  -k 'not test_reference_model_builds_neighbor_hook_without_eager_dynamics_import'

"$_baseline_python" -m pytest -q \
  packages/framework/test/dynamics/test_state_management.py -k 'nve or fire'

"$_baseline_python" -m pytest -q \
  packages/framework/test/dynamics/test_ops.py \
  -k 'VelocityVerlet or FireOps or Fire2Ops'

"$_baseline_python" -m compileall -q \
  packages/framework/nvalchemi packages/ops/nvalchemiops
git diff --check
