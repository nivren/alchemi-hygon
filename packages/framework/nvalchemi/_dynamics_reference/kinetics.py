# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-free kinetic observables for batched dynamics reference paths."""

from __future__ import annotations

import torch

KB_EV: float = 8.617333262e-5

__all__ = ["KB_EV", "kinetic_energy_per_graph", "temperature_per_graph"]


def _validate_inputs(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if velocities.ndim != 2 or velocities.shape[-1] != 3:
        raise ValueError(
            f"velocities must have shape [N, 3], got {tuple(velocities.shape)}"
        )
    if masses.ndim == 2 and masses.shape[-1] == 1:
        masses = masses.squeeze(-1)
    elif masses.ndim != 1:
        raise ValueError(
            f"masses must have shape [N] or [N, 1], got {tuple(masses.shape)}"
        )
    if masses.shape[0] != velocities.shape[0]:
        raise ValueError("masses and velocities must contain the same number of atoms")
    if batch_idx.ndim != 1 or batch_idx.shape[0] != velocities.shape[0]:
        raise ValueError("batch_idx must have shape [N]")
    if batch_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch_idx must use int32 or int64")
    if not isinstance(num_graphs, int) or num_graphs < 0:
        raise ValueError("num_graphs must be a non-negative integer")
    if batch_idx.device != velocities.device or masses.device != velocities.device:
        raise ValueError("velocities, masses, and batch_idx must share a device")
    if not velocities.is_floating_point() or masses.dtype != velocities.dtype:
        raise TypeError("velocities and masses must have the same floating-point dtype")
    if batch_idx.numel() and num_graphs == 0:
        raise ValueError("non-empty batch_idx requires num_graphs > 0")
    if batch_idx.numel() and bool(torch.any(batch_idx < 0)):
        raise ValueError("batch_idx contains a negative system index")
    if batch_idx.numel() and bool(torch.any(batch_idx >= num_graphs)):
        raise ValueError("batch_idx contains a system index outside num_graphs")
    return masses


@torch.library.custom_op("nvalchemi::reference_kinetic_energy", mutates_args=())
def kinetic_energy_per_graph(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    r"""Return ``0.5 * sum_i(m_i ||v_i||²)`` with shape ``[B, 1]``."""
    masses = _validate_inputs(velocities, masses, batch_idx, num_graphs)
    per_atom = 0.5 * masses * velocities.square().sum(dim=-1)
    result = torch.zeros(num_graphs, device=velocities.device, dtype=velocities.dtype)
    result.index_add_(0, batch_idx.to(torch.int64), per_atom)
    return result.unsqueeze(-1)


@kinetic_energy_per_graph.register_fake
def _kinetic_energy_per_graph_fake(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    return torch.empty(
        (num_graphs, 1), device=velocities.device, dtype=velocities.dtype
    )


@torch.library.custom_op("nvalchemi::reference_temperature_per_graph", mutates_args=())
def temperature_per_graph(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    atoms_per_graph: torch.Tensor,
    conversion_factor: float = KB_EV,
) -> torch.Tensor:
    """Return instantaneous temperature per graph using ``3N`` degrees of freedom."""
    if conversion_factor <= 0:
        raise ValueError("conversion_factor must be positive")
    ke = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs).squeeze(-1)
    if atoms_per_graph.ndim != 1 or atoms_per_graph.shape[0] != num_graphs:
        raise ValueError("atoms_per_graph must have shape [num_graphs]")
    if atoms_per_graph.device != velocities.device:
        raise ValueError("atoms_per_graph must share the velocity device")
    if torch.any(atoms_per_graph <= 0):
        raise ValueError("atoms_per_graph must be positive")
    n_atoms = atoms_per_graph.to(dtype=ke.dtype)
    return (2.0 * ke) / (3.0 * n_atoms * conversion_factor)


@temperature_per_graph.register_fake
def _temperature_per_graph_fake(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    atoms_per_graph: torch.Tensor,
    conversion_factor: float = KB_EV,
) -> torch.Tensor:
    return torch.empty(
        (num_graphs,), device=velocities.device, dtype=velocities.dtype
    )
