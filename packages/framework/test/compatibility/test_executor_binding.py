# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-package checks for framework-owned registered executor targets."""

from __future__ import annotations

from nvalchemiops.backend import backend_capabilities
from nvalchemiops.executor import load_entrypoint
from nvalchemiops.backend import BackendSelection


def test_framework_executor_catalog_entrypoints_are_importable() -> None:
    """Every framework-owned catalog entry must expose its declared ABI."""
    for implementation in backend_capabilities():
        if implementation.executor_owner != "framework":
            continue
        if implementation.executor is None:
            continue
        selection = BackendSelection(
            requested=implementation.implementation_id,
            implementation_id=implementation.implementation_id,
            family=implementation.family,
            strategy=implementation.strategy,
            operation=implementation.operation,
            device="cpu",
            dtype="float64",
            gradient_order=0,
            features=tuple(sorted(implementation.features)),
            reason="executor binding contract test",
        )
        for entrypoint_name in implementation.entrypoints:
            assert callable(load_entrypoint(selection, entrypoint_name))
