# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for capability-aware backend selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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
from nvalchemiops.executor import clear_entrypoint_cache, load_entrypoint


@pytest.fixture(autouse=True)
def _reset_entrypoint_cache() -> None:
    """Keep module/entrypoint replacement tests isolated from the process cache."""
    clear_entrypoint_cache()
    yield
    clear_entrypoint_cache()


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


def test_explicit_cell_list_strategy_requires_its_registered_width() -> None:
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
    periodic = resolve_backend(
        "torch_reference",
        operation="neighbor_list",
        device="cpu",
        dtype=torch.float64,
        features={"periodic", "half", "matrix"},
        strategy="cell_list",
    )
    assert periodic.implementation_id == "torch_reference.neighbor.cell_list-v1"


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
        ("hip", "cell_list"),
        ("warp", None),
    }


def test_registry_rejects_duplicate_implementation_ids() -> None:
    implementation = Implementation(
        implementation_id="test.impl-v1",
        operation="test",
        family="test",
        executor="nvalchemiops.torch_reference",
        entrypoints=("neighbor_list",),
        executor_owner="ops",
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


def test_registry_requires_executable_metadata_for_non_legacy_implementations() -> None:
    with pytest.raises(ValueError, match="executor module"):
        ImplementationRegistry(
            [
                Implementation(
                    implementation_id="test.missing-executor-v1",
                    operation="test",
                    family="test",
                    executor_owner="ops",
                    entrypoints=("run",),
                    default_strategy=True,
                )
            ]
        )

    with pytest.raises(ValueError, match="unique entrypoints"):
        ImplementationRegistry(
            [
                Implementation(
                    implementation_id="test.missing-entrypoint-v1",
                    operation="test",
                    family="test",
                    executor="nvalchemiops.torch_reference",
                    executor_owner="ops",
                    default_strategy=True,
                )
            ]
        )


def test_legacy_metadata_has_no_registry_executor() -> None:
    legacy = next(
        item
        for item in backend_capabilities()
        if item.implementation_id == "warp.legacy-upstream-v1"
    )
    assert legacy.executor is None
    assert legacy.entrypoints == ()
    assert legacy.executor_owner == "framework"


def test_registered_ops_entrypoints_are_loaded_and_called() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    neighbor_selection = resolve_backend(
        "torch_reference",
        operation="neighbor_list",
        device=positions.device,
        dtype=positions.dtype,
        features={"no_pbc", "full", "matrix"},
    )
    neighbor = load_entrypoint(neighbor_selection, "neighbor_list")
    assert neighbor is load_entrypoint(neighbor_selection, "neighbor_list")
    matrix, counts = neighbor(
        positions,
        2.0,
        batch_idx=torch.zeros(2, dtype=torch.int32),
        batch_ptr=torch.tensor([0, 2], dtype=torch.int64),
        max_neighbors=2,
        half_fill=False,
        fill_value=2,
    )
    assert matrix.shape == (2, 2)
    assert counts.tolist() == [1, 1]

    lj_selection = resolve_backend(
        "torch_reference",
        operation="lj_energy_forces",
        device=positions.device,
        dtype=positions.dtype,
        gradient_order=2,
        features={"no_pbc", "full", "forces"},
    )
    lj = load_entrypoint(lj_selection, "lj_energy_forces")
    atomic_energy, forces = lj(
        positions.requires_grad_(),
        torch.tensor([[1], [0]], dtype=torch.int32),
        torch.ones(2, dtype=torch.int32),
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
        half_list=False,
        fill_value=2,
    )
    assert atomic_energy.shape == (2,)
    assert forces.shape == (2, 3)


def test_entrypoint_load_failure_identifies_executor_package_owner() -> None:
    implementation = Implementation(
        implementation_id="test.framework.missing-executor-v1",
        operation="test_operation",
        family="test",
        strategy="default",
        executor="nvalchemiops_test_missing_framework_executor",
        entrypoints=("run",),
        executor_owner="framework",
        default_strategy=True,
    )
    registry = ImplementationRegistry([implementation])
    selection = registry.resolve(
        implementation.implementation_id,
        operation=implementation.operation,
        device="cpu",
        strategy=implementation.strategy,
    )

    with pytest.raises(BackendUnavailableError, match="framework-owned package"):
        load_entrypoint(selection, "run", registry=registry)


def test_entrypoint_cache_can_be_cleared_after_module_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "nvalchemiops_test_reloadable_executor"
    first_module = ModuleType(module_name)
    second_module = ModuleType(module_name)

    def first_run(value: int) -> str:
        return f"first:{value}"

    def second_run(value: int) -> str:
        return f"second:{value}"

    first_module.run = first_run
    second_module.run = second_run
    implementation = Implementation(
        implementation_id="test.reloadable.executor-v1",
        operation="test_operation",
        family="test",
        strategy="default",
        executor=module_name,
        entrypoints=("run",),
        executor_owner="ops",
        default_strategy=True,
    )
    registry = ImplementationRegistry([implementation])
    selection = registry.resolve(
        implementation.implementation_id,
        operation=implementation.operation,
        device="cpu",
        strategy=implementation.strategy,
    )

    monkeypatch.setitem(sys.modules, module_name, first_module)
    assert load_entrypoint(selection, "run", registry=registry)(1) == "first:1"

    monkeypatch.setitem(sys.modules, module_name, second_module)
    assert load_entrypoint(selection, "run", registry=registry)(1) == "first:1"

    clear_entrypoint_cache()
    assert load_entrypoint(selection, "run", registry=registry)(1) == "second:1"


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


def test_langevin_is_a_fixed_cell_reference_operation() -> None:
    selection = resolve_backend(
        "torch_reference",
        operation="langevin",
        device="cpu",
        dtype=torch.float64,
        features={"fixed_cell"},
    )
    assert selection.implementation_id == "torch_reference.langevin-v1"
    assert selection.operation == "langevin"


def test_default_catalog_preserves_inventory_without_importing_executors() -> None:
    """Catalog modules hold data only; importing them must not initialize Warp."""
    package_root = Path(__file__).resolve().parents[2]
    script = """
import sys
import nvalchemiops._backend_catalog as catalog
assert 'warp' not in sys.modules
assert 'nvalchemiops.torch_reference' not in sys.modules
assert [item.implementation_id for item in catalog.default_implementations()] == [
    'warp.legacy-upstream-v1',
    'torch_reference.neighbor.dense-v1',
    'torch_reference.neighbor.cell_list-v1',
    'hip.neighbor.cell_list-v1',
    'torch_reference.lj_energy_forces-v1',
    'torch_reference.velocity_verlet-v1',
    'torch_reference.fire-v1',
    'torch_reference.fire2-v1',
    'torch_reference.kinetics-v1',
    'torch_reference.langevin-v1',
    'torch_reference.periodic_wrap-v1',
    'torch_reference.segmented_reduce-v1',
]
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_root)
    subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
