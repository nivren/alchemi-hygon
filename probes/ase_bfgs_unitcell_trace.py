# SPDX-License-Identifier: Apache-2.0
"""Three M3 analytic updates for isolated HIP/LAPACK trace, no CPU oracle math."""

import argparse
import json
from pathlib import Path

import torch
from ase_bfgs_unitcell import LifecycleDeviceGuard
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from test.compatibility.test_ase_bfgs_cell import CellModel, make_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch_reference", "hip"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    data = make_batch(device)
    dynamics = ASECompatibleBFGSVariableCell(
        CellModel(), fmax=1e-10, backend=args.backend
    )
    dynamics._ensure_state_initialized(data)
    dynamics._prime(data)
    guard = LifecycleDeviceGuard(device)
    with guard:
        dynamics.run(data, n_steps=3)
    torch.cuda.synchronize()
    result = dict(
        status="passed",
        backend=args.backend,
        scalar_dofs=15,
        updates=3,
        device=str(device),
        dtype=str(data.positions.dtype),
        guard_calls=dict(guard.calls),
        numerical_tensor_residence="device-local; scalar bool/int status metadata allowed",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "guard_calls"}))


if __name__ == "__main__":
    main()
