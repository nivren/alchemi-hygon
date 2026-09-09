# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for capability-aware backend selection."""

from __future__ import annotations

import pytest
import torch

import nvalchemiops.backend as backend_module
from nvalchemiops.backend import (
    BackendAutoSelectionWarning,
    BackendUnavailableError,
    backend_capabilities,
    resolve_backend,
)


def test_explicit_neighbor_reference_requires_a_registered_width() -> None:
    selection = resolve_backend(
        "torch_reference",
        operation="neighbor_list",
        device=torch.device("cpu"),
        dtype=torch.float64,
        features={"no_pbc", "full", "matrix"},
    )
    assert selection.selected == "torch_reference"
    assert selection.features == ("full", "matrix", "no_pbc")
    assert selection.as_dict()["dtype"] == "float64"


def test_explicit_no_pbc_cell_list_requires_its_registered_width() -> None:
    """The cell-list backend is explicit and scoped to no-PBC neighbors."""
    selection = resolve_backend(
        "torch_reference_cell_list",
        operation="neighbor_list",
        device=torch.device("cpu"),
        dtype=torch.float64,
        features={"no_pbc", "full", "matrix"},
    )
    assert selection.selected == "torch_reference_cell_list"
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "torch_reference_cell_list",
            operation="neighbor_list",
            device="cpu",
            dtype=torch.float64,
            features={"periodic", "full", "matrix"},
        )


def test_periodic_half_list_is_rejected_by_the_capability_table() -> None:
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "torch_reference",
            operation="neighbor_list",
            device="cpu",
            dtype=torch.float32,
            features={"periodic", "half", "matrix"},
        )


def test_auto_warns_once_and_records_the_verified_choice() -> None:
    backend_module._AUTO_WARNED.clear()
    request = dict(
        operation="neighbor_list",
        device="cpu",
        dtype=torch.float32,
        features={"no_pbc", "full", "vectors"},
    )
    with pytest.warns(BackendAutoSelectionWarning, match="torch_reference"):
        first = resolve_backend("auto", **request)
    second = resolve_backend("auto", **request)
    assert first.selected == second.selected == "torch_reference"
    assert "highest-priority verified capability" in first.reason


def test_legacy_default_is_a_warp_selection_without_a_warp_probe() -> None:
    selection = resolve_backend(
        None,
        operation="neighbor_list",
        device="cuda:0",
        dtype=torch.float32,
        features={"no_pbc", "full", "matrix"},
    )
    assert selection.selected == "warp"
    assert "legacy upstream Warp default" in selection.reason


def test_unknown_backend_and_unregistered_optimization_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("metal", operation="neighbor_list")
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "triton",
            operation="neighbor_list",
            device="cuda",
            dtype=torch.float32,
            features={"no_pbc", "full", "matrix"},
        )


def test_capability_inventory_is_operation_scoped() -> None:
    capabilities = backend_capabilities(operation="neighbor_list")
    assert {capability.backend for capability in capabilities} == {
        "torch_reference",
        "torch_reference_cell_list",
    }
