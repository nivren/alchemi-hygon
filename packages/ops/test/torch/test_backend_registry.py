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
    Implementation,
    ImplementationRegistry,
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
    assert selection.implementation_id == "torch_reference.neighbor.dense-v1"
    assert selection.strategy == "dense"
    assert selection.features == ("full", "matrix", "no_pbc")
    assert selection.as_dict()["dtype"] == "float64"


def test_explicit_no_pbc_cell_list_strategy_requires_its_registered_width() -> None:
    """Cell-list is a neighbor strategy, not a distinct backend family."""
    selection = resolve_backend(
        "torch_reference",
        operation="neighbor_list",
        device=torch.device("cpu"),
        dtype=torch.float64,
        features={"no_pbc", "full", "matrix"},
        strategy="cell_list",
    )
    assert selection.implementation_id == "torch_reference.neighbor.cell_list-v1"
    assert selection.family == "torch_reference"
    assert selection.strategy == "cell_list"
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "torch_reference",
            operation="neighbor_list",
            device="cpu",
            dtype=torch.float64,
            features={"periodic", "full", "matrix"},
            strategy="cell_list",
        )


def test_exact_implementation_id_selects_its_registered_strategy() -> None:
    selection = resolve_backend(
        "torch_reference.neighbor.cell_list-v1",
        operation="neighbor_list",
        device="cpu",
        dtype=torch.float64,
        features={"no_pbc", "full", "matrix"},
    )
    assert selection.implementation_id == "torch_reference.neighbor.cell_list-v1"
    assert selection.family == "torch_reference"
    assert selection.strategy == "cell_list"


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
    assert first.implementation_id == "torch_reference.neighbor.dense-v1"
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
    assert selection.implementation_id == "warp.legacy-upstream-v1"
    assert selection.requested is None
    assert "legacy upstream Warp default" in selection.reason


def test_unknown_backend_and_unregistered_optimization_fail_explicitly() -> None:
    with pytest.raises(BackendUnavailableError, match="unknown backend request"):
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
    assert {(capability.family, capability.strategy) for capability in capabilities} == {
        ("torch_reference", "dense"),
        ("torch_reference", "cell_list"),
        ("warp", None),
    }


def test_registry_rejects_duplicate_implementation_ids() -> None:
    implementation = Implementation(
        implementation_id="test.impl-v1",
        operation="test",
        family="test",
        default_strategy=True,
    )
    registry = ImplementationRegistry([implementation])
    with pytest.raises(ValueError, match="duplicate implementation_id"):
        registry.register(implementation)


def test_auto_never_selects_an_operation_strategy_without_a_profile() -> None:
    with pytest.raises(BackendUnavailableError, match="does not select an operation strategy"):
        resolve_backend(
            "auto",
            operation="neighbor_list",
            device="cpu",
            dtype=torch.float32,
            features={"no_pbc", "full", "matrix"},
            strategy="cell_list",
        )


def test_fire_and_fire2_are_independent_operation_contracts() -> None:
    fire = resolve_backend(
        "torch_reference",
        operation="fire",
        device="cpu",
        dtype=torch.float64,
        gradient_order=1,
        features={"fixed_cell"},
    )
    fire2 = resolve_backend(
        "torch_reference",
        operation="fire2",
        device="cpu",
        dtype=torch.float64,
        gradient_order=1,
        features={"fixed_cell"},
    )

    assert fire.implementation_id == "torch_reference.fire-v1"
    assert fire.operation == "fire"
    assert fire2.implementation_id == "torch_reference.fire2-v1"
    assert fire2.operation == "fire2"
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "torch_reference",
            operation="fire2",
            device="cpu",
            dtype=torch.float64,
            gradient_order=1,
            features={"variable_cell"},
        )
