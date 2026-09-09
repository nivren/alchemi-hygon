# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin framework adapters for the authoritative :mod:`nvalchemiops` registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nvalchemiops.backend import BackendSelection, resolve_backend


def resolve_compute_backend(
    backend: str | None,
    *,
    operation: str,
    device: Any = None,
    dtype: Any = None,
    gradient_order: int = 0,
    features: Iterable[str] = (),
    strategy: str | None = None,
) -> BackendSelection:
    """Resolve framework compute work through the shared ops capability table."""
    return resolve_backend(
        backend,
        operation=operation,
        device=device,
        dtype=dtype,
        gradient_order=gradient_order,
        features=features,
        strategy=strategy,
    )


__all__ = ["resolve_compute_backend"]
