# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference segmented reductions used by observer hooks."""

from __future__ import annotations

from typing import Literal

import torch

ScatterReduce = Literal["amax", "sum", "amin", "mean"]


def scatter_reduce_per_graph(
    values: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    reduce: ScatterReduce = "amax",
) -> torch.Tensor:
    """Reduce a one-dimensional node tensor into graph-level values."""
    idx = batch_idx.to(dtype=torch.long)
    if reduce == "sum":
        out = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
        return out.index_add(0, idx, values)
    if reduce == "amax":
        out = torch.full(
            (num_graphs,), float("-inf"), device=values.device, dtype=values.dtype
        )
        return out.scatter_reduce(0, idx, values, reduce="amax", include_self=True)
    if reduce == "amin":
        out = torch.full(
            (num_graphs,), float("inf"), device=values.device, dtype=values.dtype
        )
        return out.scatter_reduce(0, idx, values, reduce="amin", include_self=True)
    sums = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
    sums = sums.index_add(0, idx, values)
    counts = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
    counts = counts.index_add(0, idx, torch.ones_like(values, dtype=values.dtype))
    return torch.where(counts > 0, sums / counts, torch.zeros_like(sums))


__all__ = ["scatter_reduce_per_graph"]
