#!/usr/bin/env bash
# Run the assigned-device HCU smoke batch for the Torch-reference baseline.
#
# Invoke this only from a scheduled/device-visible host terminal, for example:
#   HIP_VISIBLE_DEVICES=0 scripts/check_hcu_reference_smoke.sh
# The script never selects a device implicitly and never treats a failed HCU
# invocation as a CPU fallback.

set -euo pipefail

: "${HIP_VISIBLE_DEVICES:?set HIP_VISIBLE_DEVICES to an assigned HCU before running this smoke batch}"

_baseline_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$_baseline_root"

# shellcheck source=/dev/null
source scripts/activate_hygon_env.sh project

_baseline_python="$_baseline_root/.venv/bin/python"
export PYTHONPATH="$_baseline_root/packages/framework:$_baseline_root/packages/ops${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

timeout 90 "$_baseline_python" probes/neighbor_lj_reference.py --device cuda
timeout 90 "$_baseline_python" probes/pbc_neighbor_reference.py --device cuda
timeout 90 "$_baseline_python" probes/dynamics_reference_velocity_verlet.py --device cuda
timeout 90 "$_baseline_python" probes/dynamics_reference_kinetics.py --device cuda
timeout 90 "$_baseline_python" probes/dynamics_reference_fire.py --device cuda
