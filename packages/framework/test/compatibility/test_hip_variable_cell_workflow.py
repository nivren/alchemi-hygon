# SPDX-License-Identifier: Apache-2.0
"""Public variable-cell workflows with actual HIP COO topology and state audits."""

from collections import Counter

import pytest
import torch

from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell, FusedStage
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from test.compatibility.test_variable_cell_fire_reference import CellHarmonic, batch
from test.compatibility.test_variable_cell_workflow import convergence


DEVICES = [
    ("cpu", "torch_reference"),
    pytest.param("cuda", "hip", marks=pytest.mark.skipif(
        not torch.version.hip or not torch.cuda.is_available(), reason="requires allocated HIP device",
    )),
]


def signature(data):
    return Counter(tuple(row) for row in torch.cat(
        (data.neighbor_list.long(), data.neighbor_list_shifts.long()), -1,
    ).cpu().tolist())


class AuditedCellHarmonic(CellHarmonic):
    """Analytic stress model; audit topology on every actual model invocation."""

    def __init__(self):
        super().__init__()
        self.model_config.neighbor_config = NeighborConfig(cutoff=3.0, format=NeighborListFormat.COO)
        self.calls = []

    def __call__(self, data):
        oracle = data.clone()
        compute_neighbors(oracle, config=self.model_config.neighbor_config, backend="torch_reference")
        assert signature(data) == signature(oracle)
        e = data.neighbor_list.long()
        assert torch.equal(data.batch_idx[e[:, 0]], data.batch_idx[e[:, 1]])
        self.calls.append({
            "ptr": data.batch_ptr.cpu().tolist(),
            "sources": data.source_index.flatten().cpu().tolist() if hasattr(data, "source_index") else [],
            "edges": len(e),
        })
        return super().__call__(data)


def build(cls, backend, model=None, **kwargs):
    model = model or AuditedCellHarmonic()
    neighbor = NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE, skin=0, backend=backend,
                                method="cell_list" if backend == "hip" else None)
    hooks = [neighbor]
    if cls is FIRE2VariableCell:
        hooks.append(AlignCellHook(backend="torch_reference"))
    dynamics = cls(model=model, dt=0.04, backend="torch_reference", hooks=hooks, **kwargs)
    return dynamics, neighbor


def prime(dynamics, neighbor, data):
    dynamics._ensure_state_initialized(data)
    dynamics._call_hooks(DynamicsStage.BEFORE_STEP, data)
    neighbor(DynamicsContext(batch=data, workflow=dynamics), DynamicsStage.BEFORE_COMPUTE)
    dynamics.compute(data)


def assert_same(actual, expected, dtype):
    tolerance = 2e-12 if dtype == torch.float64 else 3e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_short_cell_trajectory_and_control_state(cls, dtype, device, backend):
    data, oracle = batch(dtype, device), batch(dtype, device)
    dynamics, neighbor = build(cls, backend)
    reference, reference_neighbor = build(cls, "torch_reference")
    prime(dynamics, neighbor, data)
    prime(reference, reference_neighbor, oracle)
    initial_cell = data.cell.clone()
    for _ in range(12):
        dynamics.step(data)
        reference.step(oracle)
        assert signature(data) == signature(oracle)
        for key in dynamics._mutable_fields:
            assert_same(getattr(data, key), getattr(oracle, key), dtype)
        for key, value in dynamics._state:
            assert_same(value, getattr(reference._state, key), dtype)
        assert bool(torch.all(torch.linalg.det(data.cell) > 0))
    assert not torch.equal(data.cell, initial_cell)
    selected = dynamics._backend_selection if cls is FIRE2VariableCell else dynamics._fire_selection
    assert selected.family == "torch_reference"
    assert neighbor.backend == backend


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_exit_freezes_fields_and_private_state(cls, dtype, device, backend):
    data = batch(dtype, device)
    dynamics, neighbor = build(cls, backend)
    prime(dynamics, neighbor, data)
    dynamics.step(data)
    data.status = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    fields = {key: getattr(data, key).clone() for key in dynamics._mutable_fields}
    state = {key: value[1].clone() for key, value in dynamics._state}
    for _ in range(2):
        dynamics.step(data)
    for key, old in fields.items():
        lo = 2 if key in ("positions", "velocities", "forces") else 1
        assert torch.equal(getattr(data, key)[lo:], old[lo:]), key
    for key, old in state.items():
        assert torch.equal(getattr(dynamics._state, key)[1], old), key


class Dataset:
    def __init__(self, dtype):
        base = batch(dtype).to_data_list()
        self.data = [base[0].clone(), base[1].clone(), base[1].clone(), base[0].clone()]
        for i, data in enumerate(self.data):
            if i != 1:  # Three equilibria; one survivor ends by the explicit test budget.
                fractions = data.positions @ torch.linalg.inv(data.cell[0])
                data.cell.copy_(torch.eye(3, dtype=dtype)[None] * 2)
                data.positions.copy_(fractions @ data.cell[0])
            data.add_system_property("source_index", torch.tensor([[i]]))

    def __len__(self):
        return len(self.data)

    def get_metadata(self, i):
        return self.data[i].num_nodes, 0

    def __getitem__(self, i):
        return self.data[i].clone(), {}


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_refill_preserves_survivor_resets_arrival_and_rebuilds_topology(cls, dtype, device, backend):
    sampler = SizeAwareSampler(Dataset(dtype), max_atoms=6, max_batch_size=2)
    data = sampler.build_initial_batch().to(device)
    dynamics, neighbor = build(cls, backend, sampler=sampler)
    prime(dynamics, neighbor, data)
    dynamics.step(data)
    survivor_id = data.system_id[1].clone()
    state = {key: value[1].clone() for key, value in dynamics._state}
    old_geometry = data.cell[1].clone(), data.positions[2:].clone()
    data.status = torch.tensor([[1], [0]], device=device)
    result = dynamics.refill_check(data, 1)
    assert result.batch_ptr.tolist() == [0, 3, 6]
    assert torch.equal(result.system_id[0], survivor_id)
    assert torch.equal(result.cell[0], old_geometry[0])
    assert torch.equal(result.positions[:3], old_geometry[1])
    fresh = dynamics._make_new_state(1, result)
    for key, old in state.items():
        assert torch.equal(getattr(dynamics._state, key)[0], old), key
        assert torch.equal(getattr(dynamics._state, key)[1:], getattr(fresh, key)), key
    prime(dynamics, neighbor, result)
    expected = CellHarmonic()(result)
    assert_same(result.stress, expected["stress"], dtype)
    dynamics.step(result)


@pytest.mark.parametrize("device,backend", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_fused_joint_exit_budget_refill_and_priming(cls, dtype, device, backend):
    model = AuditedCellHarmonic()
    joint = convergence(stress=1e-5)
    seen = set()

    class CheckPriming:
        stage = DynamicsStage.BEFORE_PRE_UPDATE
        frequency = 1

        def __call__(self, ctx, stage):
            expected = CellHarmonic()(ctx.batch)
            assert_same(ctx.batch.stress, expected["stress"], dtype)
            assert_same(ctx.batch.energy.flatten(), expected["energy"], dtype)
            seen.update(ctx.batch.source_index.flatten().cpu().tolist())

    class BudgetExit:
        stage = DynamicsStage.AFTER_STEP
        frequency = 1

        def __call__(self, ctx, stage):
            if ctx.workflow.step_count >= 2:
                selected = ctx.batch.source_index.flatten() == 1
                ctx.batch.status[selected] = 2  # Budget, not physical convergence.

    hooks = [joint, CheckPriming(), BudgetExit()]
    if cls is FIRE2VariableCell:
        hooks.insert(0, AlignCellHook(backend="torch_reference"))
    dynamics = cls(model=model, dt=0.04, backend="torch_reference",
                   hooks=hooks, convergence_hook=joint)
    neighbor = NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE, skin=0, backend=backend,
                                method="cell_list" if backend == "hip" else None)
    sink = HostMemory(capacity=4)
    sampler = SizeAwareSampler(Dataset(dtype), max_atoms=6, max_batch_size=2)
    stage = FusedStage(
        sub_stages=[(0, dynamics)], hooks=[neighbor],
        sampler=sampler,
        sinks=[sink], refill_frequency=1, device_type=device,
    )
    # Samplers return host data; run() accepts the explicitly placed Batch.
    assert stage.run(batch=sampler.build_initial_batch().to(device), n_steps=8) is None
    stored = sink.read()
    assert len(sink) == 4 and seen == set(range(4))
    assert sorted(stored.source_index.flatten().tolist()) == list(range(4))
    assert sorted(stored.system_id.flatten().tolist()) == list(range(4))
    assert stored.status.flatten()[stored.source_index.flatten() == 1].tolist() == [2]
    assert stored.status.flatten()[stored.source_index.flatten() != 1].tolist() == [1, 1, 1]
    assert dynamics.step_count == stage.step_count == 3
    assert len({tuple(call["ptr"]) for call in model.calls}) > 1
