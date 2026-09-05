#!/usr/bin/env python3
"""Torch reference probe for the first neighbor-list → LJ vertical slice.

This probe imports the Warp-independent ``nvalchemiops.torch_reference``
implementation.  The locked upstream neighbor and LJ paths remain Warp
backed, while this reference defines the semantic oracle that a later
Triton/HIP backend must match.
The initial slice is batched, non-periodic, dense neighbor-matrix output with
full and half lists.  Topology is discrete; distances, energy, and forces use
the differentiable Torch path.
"""

from __future__ import annotations

import argparse
import json

import torch
from nvalchemiops.torch_reference import lj_energy_forces, neighbor_list


def _run(device: torch.device) -> dict[str, object]:
    dtype = torch.float64
    # Two systems are deliberately placed at the same origin.  A correct
    # batch implementation must not create cross-system pairs.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    batch_ptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    epsilon, sigma, cutoff = 0.7, 1.0, 1.5

    results: dict[str, object] = {"device": str(device), "dtype": str(dtype)}
    energies: dict[str, float] = {}
    force_norms: dict[str, float] = {}
    pair_sets: dict[str, list[list[int]]] = {}
    for half_list in (False, True):
        matrix, counts = neighbor_list(
            positions,
            cutoff,
            batch_ptr=batch_ptr,
            max_neighbors=4,
            half_fill=half_list,
        )
        atomic_energies, forces = lj_energy_forces(
            positions,
            matrix,
            counts,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            half_list=half_list,
        )
        energy = atomic_energies.sum()
        key = "half" if half_list else "full"
        active_pairs = []
        for i in range(positions.shape[0]):
            for slot in range(int(counts[i].item())):
                active_pairs.append([i, int(matrix[i, slot].item())])
        pair_sets[key] = active_pairs
        energies[key] = float(energy.detach().cpu())
        force_norms[key] = float(forces.detach().norm().cpu())

        # The two systems each contain exactly one pair; topology must not
        # cross the batch boundary and half/full energy must agree.
        assert set(map(tuple, active_pairs)) == (
            {(0, 1), (1, 0), (3, 4), (4, 3)}
            if not half_list
            else {(0, 1), (3, 4)}
        )
        assert torch.allclose(forces[0] + forces[1], positions.new_zeros(3))
        assert torch.allclose(forces[3] + forces[4], positions.new_zeros(3))

    expected_pair_energy = sum(
        4.0 * epsilon * ((sigma / distance) ** 12 - (sigma / distance) ** 6)
        for distance in (1.0, 1.2)
    )
    assert abs(energies["full"] - expected_pair_energy) < 1e-12
    assert abs(energies["half"] - expected_pair_energy) < 1e-12
    assert abs(energies["full"] - energies["half"]) < 1e-12

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    results.update(
        {
            "batch_ptr": batch_ptr.tolist(),
            "pairs_full": pair_sets["full"],
            "pairs_half": pair_sets["half"],
            "energy_full": energies["full"],
            "energy_half": energies["half"],
            "energy_expected": expected_pair_energy,
            "force_norm_full": force_norms["full"],
            "force_norm_half": force_norms["half"],
            "status": "passed",
        }
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        if not (torch.version.hip and torch.cuda.is_available()):
            raise RuntimeError("HIP device required; no silent CPU fallback")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(json.dumps(_run(device), sort_keys=True))


if __name__ == "__main__":
    main()
