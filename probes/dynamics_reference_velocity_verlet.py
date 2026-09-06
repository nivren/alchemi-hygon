"""Run the Warp-free reference velocity-Verlet contract on CPU or HCU."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi._dynamics_reference import vv_position_update, vv_velocity_finalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)

    positions = torch.zeros((3, 3), dtype=torch.float64, device=device)
    velocities = torch.zeros_like(positions)
    forces = torch.tensor(
        [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    masses = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64, device=device)
    dt = torch.tensor([0.1, 0.2], dtype=torch.float64, device=device)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)

    vv_position_update(positions, velocities, forces, masses, dt, batch_idx)
    vv_velocity_finalize(velocities, -forces, masses, dt, batch_idx)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    expected_positions = torch.tensor(
        [[0.005, 0.0, 0.0], [0.005, 0.0, 0.0], [0.04, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    max_error = float((positions - expected_positions).abs().max().cpu())
    max_velocity = float(velocities.abs().max().cpu())
    if max_error > 1e-12 or max_velocity > 1e-12:
        raise AssertionError(
            f"velocity-Verlet reference mismatch: position error={max_error}, "
            f"velocity max={max_velocity}"
        )

    result = {
        "backend": "torch_reference",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(positions.dtype),
        "max_position_error": max_error,
        "max_final_velocity": max_velocity,
        "warp_imported": "warp" in __import__("sys").modules,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
