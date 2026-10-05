# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable stress-to-cell-force reference for row lattice vectors."""

from __future__ import annotations

import torch

__all__ = ["stress_to_cell_force"]


@torch.library.custom_op("nvalchemiops::stress_to_cell_force", mutates_args=())
def stress_to_cell_force(
    stress: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    keep_aligned: bool = True,
) -> torch.Tensor:
    r"""Return ``-V * inverse(cell).T @ stress`` without changing inputs.

    Parameters
    ----------
    stress : torch.Tensor
        Tensile-positive Cartesian Cauchy stress ``[B, 3, 3]``, in eV/Å³.
    cell : torch.Tensor
        Invertible row lattice matrices ``[B, 3, 3]``, in Å.
    volume : torch.Tensor
        Positive supplied cell volumes ``[B]``, in Å³. Treated as independent
        inputs by autograd; the caller supplies ``abs(det(cell))`` physically.
    keep_aligned : bool, optional
        Project onto the six lower-triangular cell entries. Default True.

    Returns
    -------
    torch.Tensor
        Fresh contiguous cell forces ``[B, 3, 3]`` in eV/Å, with the input
        float32/float64 dtype and device, including when B is zero.

    Notes
    -----
    Supports strided inputs and first/second derivatives. Uses the current
    PyTorch stream; never changes device or precision. No atom-count
    normalization, external pressure, cell alignment or cell update is applied.
    """
    if stress.ndim != 3 or stress.shape[1:] != (3, 3):
        raise ValueError("stress must have shape [B, 3, 3]")
    if cell.shape != stress.shape or volume.shape != (stress.shape[0],):
        raise ValueError("cell must match stress [B, 3, 3]; volume must be [B]")
    if stress.dtype not in (torch.float32, torch.float64):
        raise TypeError("stress/cell/volume require float32 or float64")
    if any(t.dtype != stress.dtype for t in (cell, volume)):
        raise TypeError("stress, cell and volume must have the same dtype")
    if any(t.device != stress.device for t in (cell, volume)):
        raise ValueError("stress, cell and volume must be on the same device")
    if stress.device.type not in ("cpu", "cuda"):
        raise ValueError("stress reference requires a CPU or cuda/HIP device")
    if not all(torch.isfinite(t).all() for t in (stress, cell, volume)):
        raise ValueError("stress, cell and volume must contain finite values")
    if torch.any(volume <= 0):
        raise ValueError("volume must be strictly positive")
    # CELL-STRESS-ROW / UBF-009: r=s@H implies F_H=-V H^-T sigma,
    # not -V sigma H^-T. See fork ops c4c3bb56b6c6 and the FP64 energy oracle.
    solved, info = torch.linalg.solve_ex(cell.mT, stress)
    if torch.any(info != 0):
        raise ValueError("cell must be invertible")
    result = -volume[:, None, None] * solved
    return (torch.tril(result) if keep_aligned else result).contiguous()


@stress_to_cell_force.register_fake
def _stress_to_cell_force_fake(stress, cell, volume, keep_aligned=True):
    """Produce only the shape/device/dtype contract for tracing."""
    return stress.new_empty(stress.shape)


def _setup_context(ctx, inputs, output):
    """Save original inputs so backward can itself build a derivative graph."""
    stress, cell, volume, keep_aligned = inputs
    ctx.save_for_backward(stress, cell, volume)
    ctx.keep_aligned = keep_aligned


def _backward(ctx, grad_output):
    """Differentiate the linear solve using graph-preserving Torch operations."""
    stress, cell, volume = ctx.saved_tensors
    grad = torch.tril(grad_output) if ctx.keep_aligned else grad_output
    solved = torch.linalg.solve(cell.mT, stress)
    adjoint = torch.linalg.solve(cell, grad)
    scale = volume[:, None, None]
    return (
        -scale * adjoint,
        scale * (solved @ adjoint.mT),
        -(grad * solved).sum(dim=(-2, -1)),
        None,
    )


stress_to_cell_force.register_autograd(_backward, setup_context=_setup_context)
