# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-package checks for framework-owned registered executor targets."""

from __future__ import annotations

from pathlib import Path

from nvalchemiops.backend import BackendSelection, backend_capabilities
from nvalchemiops.executor import load_entrypoint


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


def test_dispatchers_do_not_hardcode_implementation_selection() -> None:
    """Keep implementation binding in the registry and generic adapter."""
    framework_root = Path(__file__).resolve().parents[2] / "nvalchemi"
    ops_root = Path(__file__).resolve().parents[3] / "ops" / "nvalchemiops"
    dispatcher_paths = (
        framework_root / "neighbors.py",
        framework_root / "hooks" / "neighbor_list.py",
        framework_root / "hooks" / "periodic.py",
        framework_root / "models" / "lj.py",
        framework_root / "dynamics" / "_ops" / "velocity_verlet.py",
        framework_root / "dynamics" / "_ops" / "fire.py",
        framework_root / "dynamics" / "_ops" / "langevin.py",
        framework_root / "dynamics" / "hooks" / "_utils.py",
        ops_root / "dispatch.py",
        ops_root / "torch_backend.py",
    )
    forbidden_fragments = (
        "implementation_id",
        "torch_reference.",
        "warp.legacy-upstream-v1",
    )
    for path in dispatcher_paths:
        source = path.read_text()
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{path} hardcodes {fragment!r}"
