# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private, metadata-only inventory for registered backend implementations."""

from __future__ import annotations

from nvalchemiops._backend_types import Implementation

from .dynamics import IMPLEMENTATIONS as _DYNAMICS_IMPLEMENTATIONS
from .interactions import IMPLEMENTATIONS as _INTERACTION_IMPLEMENTATIONS
from .legacy import IMPLEMENTATIONS as _LEGACY_IMPLEMENTATIONS
from .neighbors import IMPLEMENTATIONS as _NEIGHBOR_IMPLEMENTATIONS
from .observability import IMPLEMENTATIONS as _OBSERVABILITY_IMPLEMENTATIONS


def default_implementations() -> tuple[Implementation, ...]:
    """Return the stable default inventory in historical registration order."""
    return (
        *_LEGACY_IMPLEMENTATIONS,
        *_NEIGHBOR_IMPLEMENTATIONS,
        *_INTERACTION_IMPLEMENTATIONS,
        *_DYNAMICS_IMPLEMENTATIONS,
        *_OBSERVABILITY_IMPLEMENTATIONS,
    )


__all__ = ["default_implementations"]
