# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference operators for the first HCU backend slice.

The functions in this module are intentionally independent of the
``nvalchemiops.torch`` Warp adapters.  They provide a correctness-first
implementation for small and medium inputs while the Triton/HIP kernels are
developed.  The public dispatcher can select this module explicitly; unsupported
features fail instead of silently falling back or dropping interactions.
"""

from __future__ import annotations

import torch


class NeighborOverflowError(RuntimeError):
    """Raised when a caller-provided neighbor matrix is too small."""


def _validate_positions(positions: torch.Tensor) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype torch.float32 or torch.float64")


def _prepare_batch_ptr(
    positions: torch.Tensor,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
) -> torch.Tensor:
    n_atoms = positions.shape[0]
    if batch_ptr is not None:
        if batch_ptr.ndim != 1 or batch_ptr.numel() < 1:
            raise ValueError("batch_ptr must be a non-empty one-dimensional tensor")
        if int(batch_ptr[0].item()) != 0 or int(batch_ptr[-1].item()) != n_atoms:
            raise ValueError("batch_ptr must start at 0 and end at num atoms")
        if torch.any(batch_ptr[1:] < batch_ptr[:-1]):
            raise ValueError("batch_ptr must be non-decreasing")
        if batch_idx is not None:
            expected = torch.repeat_interleave(
                torch.arange(batch_ptr.numel() - 1, device=batch_ptr.device),
                batch_ptr[1:] - batch_ptr[:-1],
            )
            if not torch.equal(batch_idx.to(expected.device, dtype=expected.dtype), expected):
                raise ValueError("batch_idx and batch_ptr describe different layouts")
        return batch_ptr.to(device=positions.device, dtype=torch.int64)

    if batch_idx is None:
        return torch.tensor([0, n_atoms], device=positions.device, dtype=torch.int64)
    if batch_idx.ndim != 1 or batch_idx.shape[0] != n_atoms:
        raise ValueError("batch_idx must have shape (N,)")
    if batch_idx.numel() == 0:
        return torch.zeros(1, device=positions.device, dtype=torch.int64)
    idx = batch_idx.to(device=positions.device, dtype=torch.int64)
    if torch.any(idx[1:] < idx[:-1]) or int(idx[0].item()) != 0:
        raise ValueError("batch_idx must be sorted and start at system 0")
    num_systems = int(idx[-1].item()) + 1
    counts = torch.bincount(idx, minlength=num_systems)
    return torch.cat(
        [torch.zeros(1, device=positions.device, dtype=torch.int64), counts.cumsum(0)]
    )


def neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
    target_indices: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """Build a no-PBC neighbor matrix or COO list with Torch operations.

    This first backend slice keeps topology discrete and deterministic.  It
    supports contiguous batches and full/half lists.  PBC, target rows,
    rebuild state, pair callbacks, and preallocated scratch buffers are not
    silently ignored; callers receive ``NotImplementedError``.
    """
    _validate_positions(positions)
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if cell is not None or pbc is not None:
        raise NotImplementedError("Torch reference neighbor_list currently supports no PBC")
    if target_indices is not None:
        raise NotImplementedError("Torch reference neighbor_list does not support target_indices yet")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"Torch reference neighbor_list does not support: {unsupported}")

    ptr = _prepare_batch_ptr(positions, batch_idx, batch_ptr)
    n_atoms = positions.shape[0]
    if fill_value is None:
        fill_value = n_atoms
    if max_neighbors is None:
        max_neighbors = max(
            (int(ptr[i + 1].item()) - int(ptr[i].item()) - 1 for i in range(ptr.numel() - 1)),
            default=0,
        )
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if fill_value < n_atoms:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")

    matrix = torch.full(
        (n_atoms, max_neighbors),
        int(fill_value),
        dtype=torch.int32,
        device=positions.device,
    )
    counts = torch.zeros(n_atoms, dtype=torch.int32, device=positions.device)
    distances = (
        torch.zeros(n_atoms, max_neighbors, dtype=positions.dtype, device=positions.device)
        if return_distances
        else None
    )
    vectors = (
        torch.zeros(n_atoms, max_neighbors, 3, dtype=positions.dtype, device=positions.device)
        if return_vectors
        else None
    )
    cutoff_sq = cutoff * cutoff
    for system in range(ptr.numel() - 1):
        start, end = int(ptr[system].item()), int(ptr[system + 1].item())
        for i in range(start, end):
            candidates = torch.arange(start, end, device=positions.device, dtype=torch.long)
            candidates = candidates[candidates != i]
            if half_fill:
                candidates = candidates[candidates > i]
            if candidates.numel() == 0:
                continue
            rij = positions[i].unsqueeze(0) - positions.index_select(0, candidates)
            d2 = rij.square().sum(dim=1)
            selected = candidates[(d2 < cutoff_sq) & (d2 >= 1e-10)]
            selected_d2 = d2[(d2 < cutoff_sq) & (d2 >= 1e-10)]
            count = int(selected.numel())
            if count > max_neighbors:
                raise NeighborOverflowError(
                    f"neighbor capacity {max_neighbors} is smaller than row {i} count {count}"
                )
            if count:
                matrix[i, :count] = selected.to(torch.int32)
                counts[i] = count
                if distances is not None:
                    distances[i, :count] = selected_d2.sqrt()
                if vectors is not None:
                    vectors[i, :count] = positions[i] - positions.index_select(0, selected)

    if return_neighbor_list:
        row_ids = torch.arange(n_atoms, device=positions.device, dtype=torch.int32)
        row_ids = torch.repeat_interleave(row_ids, counts.to(torch.long))
        col_ids = matrix[matrix != int(fill_value)].to(torch.int32)
        # Matrix rows are filled contiguously, so boolean flattening preserves order.
        neighbor_list = torch.stack((row_ids, col_ids), dim=0)
        neighbor_ptr = torch.cat(
            [torch.zeros(1, device=positions.device, dtype=torch.int32), counts.cumsum(0)]
        )
        output: list[torch.Tensor] = [neighbor_list, neighbor_ptr]
        if distances is not None:
            output.append(distances[matrix != int(fill_value)])
        if vectors is not None:
            output.append(vectors[matrix != int(fill_value)])
        return tuple(output)

    output = [matrix, counts]
    if distances is not None:
        output.append(distances)
    if vectors is not None:
        output.append(vectors)
    return tuple(output)


def lj_energy_forces(
    positions: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float,
    half_list: bool = False,
    fill_value: int | None = None,
    switch_width: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-atom LJ energies and differentiable forces from a matrix."""
    _validate_positions(positions)
    if not positions.requires_grad:
        raise ValueError("positions must require gradients for Torch reference forces")
    if neighbor_matrix.ndim != 2 or num_neighbors.shape != (positions.shape[0],):
        raise ValueError("neighbor_matrix and num_neighbors have incompatible shapes")
    if neighbor_matrix.shape[0] != positions.shape[0] or num_neighbors.dtype != torch.int32:
        raise ValueError("neighbor_matrix must have N rows and num_neighbors must be int32")
    if neighbor_matrix.dtype not in (torch.int32, torch.int64):
        raise TypeError("neighbor_matrix must have an integer dtype")
    if torch.any(num_neighbors < 0) or torch.any(num_neighbors > neighbor_matrix.shape[1]):
        raise ValueError("num_neighbors must lie within the neighbor matrix capacity")
    if cutoff <= 0 or epsilon < 0 or sigma <= 0:
        raise ValueError("cutoff must be positive, epsilon non-negative, sigma positive")
    if switch_width != 0.0:
        raise NotImplementedError("Torch reference LJ switching is not implemented yet")
    if fill_value is None:
        fill_value = positions.shape[0]
    if fill_value < positions.shape[0]:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")

    atomic_energies = positions.new_zeros((positions.shape[0],))
    total_energy = positions.sum() * 0.0
    weight = 1.0 if half_list else 0.5
    for i in range(positions.shape[0]):
        count = int(num_neighbors[i].item())
        if count == 0:
            continue
        neighbors = neighbor_matrix[i, :count].to(torch.long)
        if torch.any(neighbors < 0) or torch.any(neighbors >= positions.shape[0]):
            raise ValueError("neighbor_matrix contains an invalid active index")
        rij = positions[i].unsqueeze(0) - positions.index_select(0, neighbors)
        distance = torch.linalg.vector_norm(rij, dim=1)
        if torch.any(distance < 1e-5) or torch.any(distance >= cutoff):
            raise ValueError("active neighbor distances must lie in [1e-5, cutoff)")
        sr = sigma / distance
        pair_energy = 4.0 * epsilon * (sr.pow(12) - sr.pow(6))
        total_energy = total_energy + weight * pair_energy.sum()
        # Upstream assigns half of each pair energy to every endpoint.  In a
        # full list each endpoint appears in its own row (weight=1/2); in a
        # half list the reverse endpoint is added explicitly below.
        atomic_energies[i] = atomic_energies[i] + (0.5 * pair_energy).sum()
        if half_list:
            atomic_energies.index_add_(0, neighbors, 0.5 * pair_energy)

    (gradient,) = torch.autograd.grad(total_energy, positions, create_graph=True)
    return atomic_energies, -gradient


__all__ = ["NeighborOverflowError", "lj_energy_forces", "neighbor_list"]
