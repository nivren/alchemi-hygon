#!/usr/bin/env python3
"""Torch reference probe for the first neighbor-list → LJ vertical slice.

This probe deliberately does not import ``nvalchemiops``.  The locked upstream
neighbor and LJ paths initialize Warp at import time, while this reference
defines the semantic oracle that a later Triton/HIP backend must match.
The initial slice is batched, non-periodic, dense neighbor-matrix output with
full and half lists.  Topology is discrete; distances, energy, and forces use
the differentiable Torch path.
"""

from __future__ import annotations

import argparse
import json

import torch


def build_reference_neighbor_matrix(
    positions: torch.Tensor,
    batch_ptr: torch.Tensor,
    cutoff: float,
    *,
    max_neighbors: int,
    half_fill: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic dense neighbor matrix without differentiating topology."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")

    n_atoms = positions.shape[0]
    fill_value = n_atoms
    matrix = torch.full(
        (n_atoms, max_neighbors),
        fill_value,
        dtype=torch.int32,
        device=positions.device,
    )
    counts = torch.zeros(n_atoms, dtype=torch.int32, device=positions.device)

    # The reference intentionally synchronizes small integer metadata.  The
    # production backend will replace this with a device-side implementation.
    ptr = batch_ptr.detach().to(device="cpu", dtype=torch.int64).tolist()
    detached = positions.detach()
    cutoff_sq = cutoff * cutoff
    for start, end in zip(ptr[:-1], ptr[1:], strict=True):
        for i in range(start, end):
            for j in range(start, end):
                if i == j or (half_fill and i > j):
                    continue
                distance_sq = float((detached[i] - detached[j]).square().sum())
                if distance_sq >= cutoff_sq or distance_sq < 1e-10:
                    continue
                slot = int(counts[i].item())
                if slot >= max_neighbors:
                    raise ValueError("neighbor capacity overflow in reference probe")
                matrix[i, slot] = j
                counts[i] += 1
    return matrix, counts


def lj_energy_from_neighbor_matrix(
    positions: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float,
    half_list: bool,
) -> torch.Tensor:
    """Evaluate LJ energy from a dense matrix using upstream counting rules."""
    weight = 1.0 if half_list else 0.5
    energy = positions.new_zeros(())
    for i in range(positions.shape[0]):
        count = int(num_neighbors[i].item())
        if count == 0:
            continue
        neighbors = neighbor_matrix[i, :count].to(torch.long)
        rij = positions[i].unsqueeze(0) - positions.index_select(0, neighbors)
        distance = torch.linalg.vector_norm(rij, dim=1)
        sigma_over_r = sigma / distance
        pair_energy = 4.0 * epsilon * (sigma_over_r.pow(12) - sigma_over_r.pow(6))
        energy = energy + weight * pair_energy.sum()
    return energy


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
        matrix, counts = build_reference_neighbor_matrix(
            positions,
            batch_ptr,
            cutoff,
            max_neighbors=4,
            half_fill=half_list,
        )
        energy = lj_energy_from_neighbor_matrix(
            positions,
            matrix,
            counts,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            half_list=half_list,
        )
        (gradient,) = torch.autograd.grad(
            energy,
            positions,
            create_graph=True,
            retain_graph=True,
        )
        forces = -gradient
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
