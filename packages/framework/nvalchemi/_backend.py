# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin framework adapters for the authoritative :mod:`nvalchemiops` registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nvalchemiops.backend import BackendSelection, resolve_backend


def resolve_neighbor_list_backend(
    backend: str | None,
    *,
    device: Any,
    dtype: Any,
    periodic: bool,
    half_list: bool,
    matrix_output: bool,
    method: str | None = None,
) -> BackendSelection:
    """Resolve the shared runtime contract for one neighbor-list request.

    This helper is deliberately limited to the public topology contract.  It
    records PBC, full/half and MATRIX/COO requirements in the same
    :class:`BackendSelection` used by both ``compute_neighbors`` and
    ``NeighborListHook``.  Geometry, force gradients and any future native
    full-pipeline admission remain downstream capabilities; they must not be
    inferred from a topology selection.

    ``backend=None``/``"warp"`` keeps the legacy framework-owned path and
    therefore does not accept an operation strategy.  Explicit unsupported
    requests are delegated to the central registry and fail with
    ``BackendUnavailableError``; this helper never falls back silently.
    """
    return resolve_compute_backend(
        backend,
        operation="neighbor_list",
        device=device,
        dtype=dtype,
        features={
            "periodic" if periodic else "no_pbc",
            "half" if half_list else "full",
            "matrix" if matrix_output else "coo",
        },
        strategy=method if backend not in (None, "warp") else None,
    )


def resolve_compute_backend(
    backend: str | None,
    *,
    operation: str,
    device: Any = None,
    dtype: Any = None,
    gradient_order: int = 0,
    features: Iterable[str] = (),
    strategy: str | None = None,
    selection: BackendSelection | None = None,
) -> BackendSelection:
    """Resolve framework compute work through the shared ops capability table."""
    if selection is not None:
        if selection.operation != operation:
            raise ValueError(
                "pre-resolved backend selection must target operation "
                f"{operation!r}, got {selection.operation!r}"
            )
        return selection
    return resolve_backend(
        backend,
        operation=operation,
        device=device,
        dtype=dtype,
        gradient_order=gradient_order,
        features=features,
        strategy=strategy,
    )


__all__ = ["resolve_compute_backend", "resolve_neighbor_list_backend"]
