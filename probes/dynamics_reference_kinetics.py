"""Run the Warp-free batched kinetic/temperature reference on CPU or HCU."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi._dynamics_reference import kinetic_energy_per_graph, temperature_per_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = torch.device(args.device)

    velocities = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    masses = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64, device=device)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    atoms_per_graph = torch.tensor([2, 1], dtype=torch.int64, device=device)

    kinetic = kinetic_energy_per_graph(velocities, masses, batch_idx, 2)
    temperature = temperature_per_graph(
        velocities, masses, batch_idx, 2, atoms_per_graph
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "backend": "torch_reference",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "kinetic_energy": kinetic.cpu().reshape(-1).tolist(),
        "temperature": temperature.cpu().tolist(),
        "warp_imported": "warp" in __import__("sys").modules,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
