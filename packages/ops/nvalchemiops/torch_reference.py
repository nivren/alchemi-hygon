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

import math

import torch


class NeighborOverflowError(RuntimeError):
    """Raised when a caller-provided neighbor matrix is too small."""


def _normalize_periodic_geometry(
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    *,
    num_systems: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cell is None or pbc is None:
        raise ValueError("cell and pbc must be provided together for periodic neighbors")
    if cell.ndim == 2:
        if num_systems != 1:
            raise ValueError("a single cell is only valid for a single-system batch")
        cells = cell.unsqueeze(0)
    elif cell.ndim == 3 and cell.shape[0] == num_systems:
        cells = cell
    else:
        raise ValueError(
            f"cell must have shape (3, 3) or ({num_systems}, 3, 3), got {tuple(cell.shape)}"
        )
    if pbc.ndim == 1:
        if num_systems != 1 or pbc.shape[0] != 3:
            raise ValueError("a single pbc row is only valid for a single-system batch")
        periodic = pbc.unsqueeze(0)
    elif pbc.ndim == 2 and pbc.shape == (num_systems, 3):
        periodic = pbc
    else:
        raise ValueError(
            f"pbc must have shape (3,) or ({num_systems}, 3), got {tuple(pbc.shape)}"
        )
    cells = cells.to(device=device, dtype=dtype)
    periodic = periodic.to(device=device, dtype=torch.bool)
    if not torch.isfinite(cells).all():
        raise ValueError("cell must contain finite values")
    return cells, periodic


def _periodic_neighbor_pairs(
    positions: torch.Tensor,
    ptr: torch.Tensor,
    cutoff: float,
    cells: torch.Tensor,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate periodic neighbors and return flat device tensors.

    The shift convention matches the upstream matrix contract:
    ``r_ij = r_i - r_j - shift @ cell``.  This reference implementation is
    intentionally eager and small-input oriented.  Each system is vectorized
    over atom pairs and image shifts, and the result remains a flat set of
    device tensors so the caller can assemble the matrix without a Python
    loop over individual edges.  Later Triton/HIP paths will replace this
    reference implementation for larger systems after the contract is
    validated.

    The returned tensors are ordered by system, then source row, destination
    atom, and image shift, matching ``torch.nonzero``'s row-major order and
    the ordering of the former Python list-of-lists implementation.
    """
    row_parts: list[torch.Tensor] = []
    col_parts: list[torch.Tensor] = []
    shift_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    vector_parts: list[torch.Tensor] = []
    cutoff_sq = cutoff * cutoff
    # ``ptr`` lives on the target device.  Read it once for the unavoidable
    # per-system Python loop instead of synchronizing once per boundary.
    ptr_cpu = ptr.detach().cpu().tolist()
    for system in range(ptr.numel() - 1):
        start = int(ptr_cpu[system])
        end = int(ptr_cpu[system + 1])
        cell_s = cells[system]
        pbc_s = pbc[system]
        try:
            inv_cell = torch.linalg.inv(cell_s)
        except RuntimeError as exc:
            raise ValueError("periodic cell must be invertible") from exc
        bounds = torch.ceil(
            cutoff * torch.linalg.vector_norm(inv_cell, dim=0)
        ).to(torch.int64) + 1
        local_positions = positions[start:end]
        delta = local_positions[:, None, :] - local_positions[None, :, :]
        fractional_delta = delta @ inv_cell
        bounds_cpu = bounds.detach().cpu().tolist()
        ranges: list[range] = []
        for dim in range(3):
            if bool(pbc_s[dim]):
                # Use one range covering every pair.  For each pair the old
                # loop used floor(fractional_delta) +/- bound; taking the
                # extrema preserves that candidate set without per-pair .item().
                lower = math.floor(float(fractional_delta[..., dim].amin().item()))
                upper = math.floor(float(fractional_delta[..., dim].amax().item()))
                lower -= int(bounds_cpu[dim]) + 1
                upper += int(bounds_cpu[dim]) + 1
                ranges.append(range(lower, upper + 1))
            else:
                ranges.append(range(0, 1))

        shifts = [
            (sx, sy, sz)
            for sx in ranges[0]
            for sy in ranges[1]
            for sz in ranges[2]
        ]
        shift_values = torch.tensor(
            shifts, dtype=torch.int32, device=positions.device
        )
        shift_vectors = shift_values.to(dtype=positions.dtype) @ cell_s
        vectors = delta[:, :, None, :] - shift_vectors[None, None, :, :]
        distance_sq = vectors.square().sum(dim=-1)
        active = distance_sq < cutoff_sq
        zero_shift = shifts.index((0, 0, 0))
        diagonal = torch.arange(end - start, device=positions.device)
        active[diagonal, diagonal, zero_shift] = False
        if torch.any(active & (distance_sq == 0)):
            raise ValueError(
                "periodic neighbor list contains an overlapping active pair"
            )

        active_indices = torch.nonzero(active, as_tuple=False)
        if active_indices.numel() == 0:
            continue

        # ``active_indices`` is already row-major.  Keep all pair metadata on
        # the target device; the matrix writer below computes row-local ranks
        # and performs one scatter per output field.
        row_parts.append(active_indices[:, 0].to(torch.long) + start)
        col_parts.append(active_indices[:, 1].to(torch.long) + start)
        shift_parts.append(shift_values.index_select(0, active_indices[:, 2]))
        distance_parts.append(distance_sq[active])
        vector_parts.append(vectors[active])

    empty_rows = torch.empty(0, dtype=torch.long, device=positions.device)
    empty_shifts = torch.empty(
        0, 3, dtype=torch.int32, device=positions.device
    )
    empty_distances = torch.empty(0, dtype=positions.dtype, device=positions.device)
    empty_vectors = torch.empty(
        0, 3, dtype=positions.dtype, device=positions.device
    )
    return (
        torch.cat(row_parts) if row_parts else empty_rows,
        torch.cat(col_parts) if col_parts else empty_rows,
        torch.cat(shift_parts) if shift_parts else empty_shifts,
        torch.cat(distance_parts) if distance_parts else empty_distances,
        torch.cat(vector_parts) if vector_parts else empty_vectors,
    )


def _neighbor_list_periodic(
    positions: torch.Tensor,
    cutoff: float,
    *,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    max_neighbors: int | None,
    fill_value: int,
    return_neighbor_list: bool,
    return_distances: bool,
    return_vectors: bool,
) -> tuple[torch.Tensor, ...]:
    if return_neighbor_list and (return_distances or return_vectors):
        raise NotImplementedError(
            "Torch reference periodic COO pair geometry is not implemented yet"
        )
    ptr = _prepare_batch_ptr(positions, batch_idx, batch_ptr)
    cells, periodic = _normalize_periodic_geometry(
        cell,
        pbc,
        num_systems=ptr.numel() - 1,
        device=positions.device,
        dtype=positions.dtype,
    )
    row_ids, col_ids, pair_shifts, pair_distance_sq, pair_vectors = (
        _periodic_neighbor_pairs(positions, ptr, cutoff, cells, periodic)
    )
    counts = torch.bincount(row_ids, minlength=positions.shape[0]).to(torch.int32)
    found_max = int(counts.max().item()) if counts.numel() else 0
    if max_neighbors is None:
        max_neighbors = found_max
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if found_max > max_neighbors:
        raise NeighborOverflowError(
            f"neighbor capacity {max_neighbors} is smaller than a periodic row count {found_max}"
        )
    matrix = torch.full(
        (positions.shape[0], max_neighbors),
        int(fill_value),
        dtype=torch.int32,
        device=positions.device,
    )
    shifts = torch.zeros(
        positions.shape[0], max_neighbors, 3, dtype=torch.int32, device=positions.device
    )
    distances = (
        torch.zeros(positions.shape[0], max_neighbors, dtype=positions.dtype, device=positions.device)
        if return_distances
        else None
    )
    vectors = (
        torch.zeros(positions.shape[0], max_neighbors, 3, dtype=positions.dtype, device=positions.device)
        if return_vectors
        else None
    )
    if row_ids.numel():
        row_starts = torch.cumsum(counts.to(torch.long), dim=0) - counts.to(torch.long)
        ranks = torch.arange(row_ids.numel(), device=positions.device) - row_starts[row_ids]
        matrix[row_ids, ranks] = col_ids.to(torch.int32)
        shifts[row_ids, ranks] = pair_shifts
        if distances is not None:
            distances[row_ids, ranks] = pair_distance_sq.sqrt()
        if vectors is not None:
            vectors[row_ids, ranks] = pair_vectors

    if return_neighbor_list:
        active = torch.arange(max_neighbors, device=positions.device).unsqueeze(0) < counts.to(
            torch.long
        ).unsqueeze(1)
        row_ids = torch.arange(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
        row_ids = torch.repeat_interleave(row_ids, counts.to(torch.long))
        edges = torch.stack((row_ids, matrix[active].to(torch.int32)), dim=0)
        ptr_out = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=positions.device), counts.cumsum(0)]
        )
        output: list[torch.Tensor] = [edges, ptr_out, shifts[active]]
        return tuple(output)

    output = [matrix, counts, shifts]
    if distances is not None:
        output.append(distances)
    if vectors is not None:
        output.append(vectors)
    return tuple(output)


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
    """Build a reference neighbor matrix or COO list with Torch operations.

    This first backend slice keeps topology discrete and deterministic.  It
    supports contiguous batches and full/half no-PBC lists.  Periodic lists
    support full lists and return integer image shifts; periodic half lists,
    target rows, rebuild state, pair callbacks, and preallocated scratch
    buffers fail explicitly.
    """
    _validate_positions(positions)
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if target_indices is not None:
        raise NotImplementedError("Torch reference neighbor_list does not support target_indices yet")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"Torch reference neighbor_list does not support: {unsupported}")

    n_atoms = positions.shape[0]
    if fill_value is None:
        fill_value = n_atoms
    if fill_value < n_atoms:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")
    if cell is not None or pbc is not None:
        if half_fill:
            raise NotImplementedError(
                "Torch reference periodic neighbor_list currently requires half_fill=False"
            )
        return _neighbor_list_periodic(
            positions,
            cutoff,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors,
            fill_value=fill_value,
            return_neighbor_list=return_neighbor_list,
            return_distances=return_distances,
            return_vectors=return_vectors,
        )

    ptr = _prepare_batch_ptr(positions, batch_idx, batch_ptr)
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
        local_positions = positions[start:end]
        local_count = end - start
        if local_count == 0:
            continue

        # ``nonzero`` is row-major, matching the upstream row fill order:
        # every source row is visited in ascending order and each target is
        # ascending within the row.  All pair discovery and output writes stay
        # on the tensor device; only batch-system boundaries remain eager.
        vectors_local = (
            local_positions[:, None, :] - local_positions[None, :, :]
        )
        distance_sq = vectors_local.square().sum(dim=-1)
        active = distance_sq < cutoff_sq
        diagonal = torch.eye(local_count, dtype=torch.bool, device=positions.device)
        active = active & ~diagonal
        if half_fill:
            active = active & torch.triu(
                torch.ones_like(diagonal), diagonal=1
            )
        if bool(torch.any(distance_sq[active] == 0)):
            raise ValueError("neighbor list contains an overlapping active pair")

        active_indices = torch.nonzero(active, as_tuple=False)
        local_rows = active_indices[:, 0]
        local_columns = active_indices[:, 1]
        local_counts = torch.bincount(local_rows, minlength=local_count)
        if bool(torch.any(local_counts > max_neighbors)):
            overflow_row = int(torch.nonzero(local_counts > max_neighbors)[0].item())
            overflow_count = int(local_counts[overflow_row].item())
            raise NeighborOverflowError(
                f"neighbor capacity {max_neighbors} is smaller than row "
                f"{start + overflow_row} count {overflow_count}"
            )
        counts[start:end].copy_(local_counts.to(torch.int32))
        if active_indices.numel() == 0:
            continue

        offsets = torch.cat(
            [
                torch.zeros(1, dtype=local_counts.dtype, device=positions.device),
                local_counts.cumsum(0)[:-1],
            ]
        )
        ranks = torch.arange(
            active_indices.shape[0], device=positions.device, dtype=local_counts.dtype
        ) - offsets.index_select(0, local_rows)
        global_rows = local_rows + start
        matrix[global_rows, ranks] = (local_columns + start).to(torch.int32)
        if distances is not None:
            distances[global_rows, ranks] = distance_sq[local_rows, local_columns].sqrt()
        if vectors is not None:
            vectors[global_rows, ranks] = vectors_local[local_rows, local_columns]

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
    cell: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-atom LJ energies and differentiable forces from a matrix.

    When ``cell`` and ``neighbor_matrix_shifts`` are supplied, the active
    pair displacement follows the framework's periodic convention
    ``r_ij = r_i - r_j - shift @ cell``.  This reference path intentionally
    supports only full periodic lists; periodic half-list ownership, switching
    and virial/stress remain outside this slice.
    """
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
    if half_list and neighbor_matrix_shifts is not None:
        raise NotImplementedError(
            "Torch reference periodic LJ currently requires half_list=False"
        )
    if fill_value is None:
        fill_value = positions.shape[0]
    if fill_value < positions.shape[0]:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")

    cells: torch.Tensor | None = None
    atom_systems: torch.Tensor | None = None
    if cell is not None:
        if cell.ndim == 2:
            cells = cell.unsqueeze(0)
        elif cell.ndim == 3:
            cells = cell
        else:
            raise ValueError(
                f"cell must have shape (3, 3) or (B, 3, 3), got {tuple(cell.shape)}"
            )
        if cells.shape[1:] != (3, 3):
            raise ValueError(
                f"cell must have shape (3, 3) or (B, 3, 3), got {tuple(cell.shape)}"
            )
        cells = cells.to(dtype=positions.dtype, device=positions.device)
        if not torch.isfinite(cells).all():
            raise ValueError("cell must contain finite values")
        try:
            torch.linalg.inv(cells)
        except RuntimeError as exc:
            raise ValueError("periodic cell must be invertible") from exc
        if batch_idx is None:
            if cells.shape[0] != 1:
                raise ValueError("batch_idx is required when cell has more than one system")
            atom_systems = torch.zeros(
                positions.shape[0], dtype=torch.long, device=positions.device
            )
        else:
            if batch_idx.shape != (positions.shape[0],):
                raise ValueError("batch_idx must have shape (N,)")
            if batch_idx.dtype not in (torch.int32, torch.int64):
                raise TypeError("batch_idx must have an integer dtype")
            atom_systems = batch_idx.to(device=positions.device, dtype=torch.long)
            if torch.any(atom_systems < 0) or torch.any(atom_systems >= cells.shape[0]):
                raise ValueError("batch_idx contains an invalid system index")
    elif batch_idx is not None:
        if batch_idx.shape != (positions.shape[0],):
            raise ValueError("batch_idx must have shape (N,)")

    if neighbor_matrix_shifts is not None:
        if neighbor_matrix_shifts.shape != (*neighbor_matrix.shape, 3):
            raise ValueError("neighbor_matrix_shifts must have shape (N, K, 3)")
        if neighbor_matrix_shifts.dtype not in (torch.int32, torch.int64):
            raise TypeError("neighbor_matrix_shifts must have an integer dtype")
        neighbor_matrix_shifts = neighbor_matrix_shifts.to(
            device=positions.device, dtype=torch.int32
        )
        if cells is None and torch.any(neighbor_matrix_shifts != 0):
            raise ValueError("cell is required when neighbor_matrix_shifts are nonzero")

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
        if atom_systems is not None:
            if torch.any(atom_systems.index_select(0, neighbors) != atom_systems[i]):
                raise ValueError("active neighbor pairs must remain within one system")
        if neighbor_matrix_shifts is not None:
            if cells is None:
                raise AssertionError("validated nonzero shifts require a periodic cell")
            shifts = neighbor_matrix_shifts[i, :count].to(dtype=positions.dtype)
            rij = rij - shifts @ cells[atom_systems[i]]
        distance = torch.linalg.vector_norm(rij, dim=1)
        active = distance < cutoff
        if torch.any(distance[active] < 1e-5):
            raise ValueError("active neighbor distances must be at least 1e-5")
        if not torch.any(active):
            continue
        active_neighbors = neighbors[active]
        active_distance = distance[active]
        sr = sigma / active_distance
        pair_energy = 4.0 * epsilon * (sr.pow(12) - sr.pow(6))
        total_energy = total_energy + weight * pair_energy.sum()
        # Upstream assigns half of each pair energy to every endpoint.  In a
        # full list each endpoint appears in its own row (weight=1/2); in a
        # half list the reverse endpoint is added explicitly below.
        atomic_energies[i] = atomic_energies[i] + (0.5 * pair_energy).sum()
        if half_list:
            atomic_energies.index_add_(0, active_neighbors, 0.5 * pair_energy)

    (gradient,) = torch.autograd.grad(total_energy, positions, create_graph=True)
    return atomic_energies, -gradient


__all__ = ["NeighborOverflowError", "lj_energy_forces", "neighbor_list"]
