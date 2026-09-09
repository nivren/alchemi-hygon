# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M1 contracts for framework-side backend resolution and dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nvalchemiops.backend import BackendSelection, resolve_backend


def _selection(
    operation: str,
    features: set[str],
    *,
    gradient_order: int = 0,
) -> BackendSelection:
    return resolve_backend(
        "torch_reference",
        operation=operation,
        device="cpu",
        dtype=torch.float64,
        gradient_order=gradient_order,
        features=features,
    )


def _fail_if_resolved_again(*args: object, **kwargs: object) -> BackendSelection:
    raise AssertionError("a pre-resolved selection must bypass registry resolution")


def test_lj_dispatcher_consumes_framework_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import nvalchemiops.torch_backend as torch_backend

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64
    ).requires_grad_()
    selection = _selection(
        "lj_energy_forces", {"no_pbc", "full", "forces"}, gradient_order=2
    )
    monkeypatch.setattr(torch_backend, "resolve_backend", _fail_if_resolved_again)

    (atomic, forces), returned = torch_backend.dispatch_lj_energy_forces(
        positions,
        torch.tensor([[1], [0]], dtype=torch.int32),
        torch.ones(2, dtype=torch.int32),
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
        selection=selection,
        return_backend=True,
    )

    assert returned is selection
    assert atomic.shape == (2,)
    assert forces.shape == (2, 3)


def test_dynamics_and_periodic_dispatchers_consume_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nvalchemi.dynamics._ops.fire as fire_module
    import nvalchemi.dynamics._ops.velocity_verlet as vv_module
    import nvalchemi.dynamics.hooks._utils as utils_module
    import nvalchemi.hooks.periodic as periodic_module
    import nvalchemi._backend as framework_backend

    vv_selection = _selection("velocity_verlet", {"fixed_cell"}, gradient_order=1)
    monkeypatch.setattr(
        framework_backend, "resolve_backend", _fail_if_resolved_again
    )
    positions = torch.zeros((2, 3), dtype=torch.float64)
    velocities = torch.ones_like(positions)
    forces = torch.ones_like(positions)
    masses = torch.ones(2, dtype=torch.float64)
    dt = torch.full((1,), 0.01, dtype=torch.float64)
    batch_idx = torch.zeros(2, dtype=torch.int32)
    vv_module.vv_position_update(
        positions,
        velocities,
        forces,
        masses,
        dt,
        batch_idx,
        selection=vv_selection,
    )

    fire_selection = _selection("fire", {"fixed_cell"}, gradient_order=1)
    fire2_selection = _selection("fire2", {"fixed_cell"}, gradient_order=1)
    fire_kwargs = dict(
        alpha=torch.full((1,), 0.1, dtype=torch.float64),
        dt=torch.full((1,), 0.01, dtype=torch.float64),
        n_steps_positive=torch.zeros(1, dtype=torch.int32),
        alpha_start=torch.full((1,), 0.1, dtype=torch.float64),
        f_alpha=torch.full((1,), 0.99, dtype=torch.float64),
        dt_min=torch.full((1,), 0.001, dtype=torch.float64),
        dt_max=torch.full((1,), 0.1, dtype=torch.float64),
        maxstep=torch.full((1,), 0.1, dtype=torch.float64),
        n_min=torch.full((1,), 5, dtype=torch.int32),
        f_dec=torch.full((1,), 0.5, dtype=torch.float64),
        f_inc=torch.full((1,), 1.1, dtype=torch.float64),
        uphill_flag=torch.zeros(1, dtype=torch.int32),
        batch_idx=batch_idx,
    )
    fire_module.fire_step(
        positions.clone(),
        velocities.clone(),
        forces,
        masses,
        selection=fire_selection,
        **fire_kwargs,
    )
    fire_module.fire2_step_coord(
        positions.clone(),
        velocities.clone(),
        forces,
        batch_idx,
        fire_kwargs["alpha"].clone(),
        fire_kwargs["dt"].clone(),
        fire_kwargs["n_steps_positive"].clone(),
        selection=fire2_selection,
    )

    kinetics_selection = _selection("kinetics", {"per_graph"}, gradient_order=1)
    reduction_selection = _selection("segmented_reduce", {"per_graph"})
    utils_module.kinetic_energy_per_graph(
        velocities,
        masses,
        batch_idx,
        num_graphs=1,
        selection=kinetics_selection,
    )
    utils_module.temperature_per_graph(
        velocities,
        masses,
        batch_idx,
        num_graphs=1,
        atoms_per_graph=torch.ones(1, dtype=torch.int64),
        selection=kinetics_selection,
    )
    utils_module.scatter_reduce_per_graph(
        torch.ones(2, dtype=torch.float64),
        batch_idx,
        num_graphs=1,
        selection=reduction_selection,
    )

    periodic_selection = _selection("periodic_wrap", {"inplace", "periodic"})
    periodic_module.wrap_positions_into_cell(
        positions.clone(),
        torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
        torch.ones((1, 3), dtype=torch.bool),
        batch_idx,
        selection=periodic_selection,
    )


def test_observer_dispatchers_pass_their_pre_resolved_selections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    logging_module = importlib.import_module("nvalchemi.dynamics.hooks.logging")
    utils_module = importlib.import_module("nvalchemi.dynamics.hooks._utils")
    framework_backend = importlib.import_module("nvalchemi._backend")

    reductions = _selection("segmented_reduce", {"per_graph"})
    kinetics = _selection("kinetics", {"per_graph"}, gradient_order=1)
    calls: list[str] = []

    def resolve_for_observer(
        backend: str | None,
        *,
        operation: str,
        **kwargs: object,
    ) -> BackendSelection:
        del backend, kwargs
        calls.append(operation)
        return {"segmented_reduce": reductions, "kinetics": kinetics}[operation]

    monkeypatch.setattr(logging_module, "resolve_compute_backend", resolve_for_observer)
    monkeypatch.setattr(
        framework_backend, "resolve_backend", _fail_if_resolved_again
    )

    batch = SimpleNamespace(
        device=torch.device("cpu"),
        num_graphs=1,
        energy=torch.zeros(1, 1, dtype=torch.float64),
        forces=torch.ones(2, 3, dtype=torch.float64),
        velocities=torch.ones(2, 3, dtype=torch.float64),
        atomic_masses=torch.ones(2, dtype=torch.float64),
        batch_idx=torch.zeros(2, dtype=torch.int32),
        num_nodes_per_graph=torch.ones(1, dtype=torch.int64) * 2,
        status=None,
    )
    ctx = SimpleNamespace(workflow=SimpleNamespace(backend="torch_reference"))
    hook = logging_module.LoggingHook(
        backend="custom", writer_fn=lambda step, rows: None, compute_backend="auto"
    )
    columns = hook._compute_columns(batch, step_count=0, ctx=ctx)

    assert columns.get("fmax").shape == (1,)
    assert columns.get("temperature").shape == (1,)
    assert calls == ["segmented_reduce", "kinetics"]
