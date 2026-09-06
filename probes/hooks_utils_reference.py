#!/usr/bin/env python3
"""Probe Warp-free observer/periodic helpers on CPU or Hygon Torch devices."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi.dynamics.hooks._utils import (
    kinetic_energy_per_graph,
    scatter_reduce_per_graph,
    temperature_per_graph,
)
from nvalchemi.hooks.periodic import wrap_positions_into_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"stage=begin device={device}", flush=True)
    values = torch.tensor([3.0, 1.0, 2.0, -1.0], device=device)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
    print("stage=reductions", flush=True)
    reductions = {}
    for reduce in ("sum", "amax", "amin", "mean"):
        result = scatter_reduce_per_graph(
            values, batch_idx, num_graphs=2, reduce=reduce, backend="torch_reference"
        )
        reductions[reduce] = result.detach().cpu().tolist()

    velocities = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]],
        device=device,
    )
    masses = torch.tensor([2.0, 1.0, 4.0, 3.0], device=device)
    print("stage=kinetics", flush=True)
    ke = kinetic_energy_per_graph(
        velocities, masses, batch_idx, num_graphs=2, backend="torch_reference"
    )
    temperature = temperature_per_graph(
        velocities,
        masses,
        batch_idx,
        num_graphs=2,
        atoms_per_graph=torch.tensor([2, 2], device=device),
        backend="torch_reference",
    )

    cell = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
         [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]],
        device=device,
    )
    pbc = torch.ones((2, 3), dtype=torch.bool, device=device)
    positions = torch.tensor(
        [[12.0, -3.0, 25.0], [7.0, 2.5, 2.5], [16.0, 5.0, 5.0], [1.0, 1.0, 1.0]],
        device=device,
    )
    print("stage=periodic_wrap", flush=True)
    wrapped = wrap_positions_into_cell(
        positions, cell, pbc, batch_idx, backend="torch_reference"
    )
    expected = torch.tensor(
        [[2.0, 7.0, 5.0], [7.0, 2.5, 2.5], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        device=device,
    )
    torch.testing.assert_close(wrapped, expected, atol=1e-5, rtol=1e-5)
    print("stage=complete", flush=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "torch_version": torch.__version__,
                "backend": "torch_reference",
                "reductions": reductions,
                "kinetic_energy": ke.detach().cpu().tolist(),
                "temperature": temperature.detach().cpu().tolist(),
                "wrapped_positions": wrapped.detach().cpu().tolist(),
                "warp_loaded": "warp" in __import__("sys").modules,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
