# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Guard the no-override path against accidental reference fallback."""

from __future__ import annotations

import importlib.util

import pytest

from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.base import NeighborConfig


@pytest.mark.xfail(
    importlib.util.find_spec("warp") is None,
    strict=True,
    reason="WARP-DEFAULT-001: verify the legacy default on an NVIDIA/Warp runner",
)
def test_neighbor_hook_none_remains_the_legacy_warp_default() -> None:
    """A missing explicit backend must never construct a Torch reference hook."""
    hook = NeighborListHook(NeighborConfig(cutoff=2.0))
    assert hook.backend is None
