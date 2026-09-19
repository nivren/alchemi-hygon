#!/usr/bin/env python3
"""Bounded CPU/HCU smoke for the fixed-cell Torch Langevin reference."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi.dynamics._ops.langevin import langevin_half_step
from nvalchemi._dynamics_reference.velocity_verlet import vv_position_update


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float64

    positions = torch.arange(21, dtype=dtype, device=device).reshape(7, 3) / 10.0
    velocities = torch.ones((7, 3), dtype=dtype, device=device) * 0.2
    forces = torch.arange(21, dtype=dtype, device=device).reshape(7, 3) / 7.0
    masses = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0], dtype=dtype, device=device)
    batch_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)
    dt = torch.tensor([0.1, 0.2], dtype=dtype, device=device)
    zero = torch.zeros(2, dtype=dtype, device=device)

    vv_positions = positions.clone()
    vv_velocities = velocities.clone()
    vv_position_update(vv_positions, vv_velocities, forces, masses, dt, batch_idx)
    langevin_positions = positions.clone()
    langevin_velocities = velocities.clone()
    langevin_half_step(
        langevin_positions,
        langevin_velocities,
        forces,
        masses,
        dt,
        zero,
        zero,
        7,
        batch_idx,
        backend="torch_reference",
    )
    torch.testing.assert_close(langevin_positions, vv_positions)
    torch.testing.assert_close(langevin_velocities, vv_velocities)

    temperature = torch.tensor([0.1, 0.2], dtype=dtype, device=device)
    friction = torch.tensor([0.5, 1.0], dtype=dtype, device=device)
    seeded_a = positions.clone()
    seeded_v_a = velocities.clone()
    seeded_b = positions.clone()
    seeded_v_b = velocities.clone()
    langevin_half_step(
        seeded_a,
        seeded_v_a,
        forces,
        masses,
        dt,
        temperature,
        friction,
        123,
        batch_idx,
        backend="torch_reference",
    )
    langevin_half_step(
        seeded_b,
        seeded_v_b,
        forces,
        masses,
        dt,
        temperature,
        friction,
        123,
        batch_idx,
        backend="torch_reference",
    )
    torch.testing.assert_close(seeded_a, seeded_b)
    torch.testing.assert_close(seeded_v_a, seeded_v_b)
    _sync(device)

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "num_atoms": positions.shape[0],
                "num_systems": dt.shape[0],
                "friction_zero_matches_vv": True,
                "seed_reproducible": True,
                "finite": bool(torch.isfinite(seeded_a).all()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
