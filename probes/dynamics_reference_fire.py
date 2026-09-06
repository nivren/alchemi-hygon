"""Run fixed-cell Torch reference FIRE and FIRE2 on CPU or HCU."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi._dynamics_reference import fire2_step_coord, fire_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float64

    positions = torch.zeros((4, 3), dtype=dtype, device=device)
    velocities = torch.ones_like(positions)
    forces = torch.ones_like(positions)
    masses = torch.ones(4, dtype=dtype, device=device)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    fire_params = {
        "alpha": torch.full((2,), 0.05, dtype=dtype, device=device),
        "dt": torch.full((2,), 0.01, dtype=dtype, device=device),
        "n_steps_positive": torch.full((2,), 10, dtype=torch.int32, device=device),
        "alpha_start": torch.full((2,), 0.1, dtype=dtype, device=device),
        "f_alpha": torch.full((2,), 0.99, dtype=dtype, device=device),
        "dt_min": torch.full((2,), 0.001, dtype=dtype, device=device),
        "dt_max": torch.full((2,), 0.1, dtype=dtype, device=device),
        "maxstep": torch.full((2,), 0.1, dtype=dtype, device=device),
        "n_min": torch.full((2,), 5, dtype=torch.int32, device=device),
        "f_dec": torch.full((2,), 0.5, dtype=dtype, device=device),
        "f_inc": torch.full((2,), 1.1, dtype=dtype, device=device),
        "uphill_flag": torch.zeros(2, dtype=torch.int32, device=device),
    }
    fire_step(
        positions,
        velocities,
        forces,
        masses,
        batch_idx=batch_idx,
        **fire_params,
    )

    fire2_positions = torch.zeros_like(positions)
    fire2_velocities = torch.ones_like(positions)
    fire2_alpha = torch.full((2,), 0.05, dtype=dtype, device=device)
    fire2_dt = torch.full((2,), 0.04, dtype=dtype, device=device)
    fire2_nsteps = torch.full((2,), 1, dtype=torch.int32, device=device)
    fire2_step_coord(
        fire2_positions,
        fire2_velocities,
        forces,
        batch_idx,
        fire2_alpha,
        fire2_dt,
        fire2_nsteps,
        delaystep=2,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not torch.isfinite(positions).all() or not torch.isfinite(fire2_positions).all():
        raise AssertionError("FIRE reference produced non-finite state")
    print(
        json.dumps(
            {
                "backend": "torch_reference",
                "device": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu",
                "fire_positions_norm": float(positions.norm().cpu()),
                "fire2_positions_norm": float(fire2_positions.norm().cpu()),
                "fire2_dt": fire2_dt.cpu().tolist(),
                "fire2_nsteps": fire2_nsteps.cpu().tolist(),
                "warp_imported": "warp" in __import__("sys").modules,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
