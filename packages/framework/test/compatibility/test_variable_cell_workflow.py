# SPDX-License-Identifier: Apache-2.0
"""Variable-cell lifecycle regressions using native dynamics and samplers."""

from itertools import product

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell, FusedStage
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.models.base import NeighborConfig
from test.compatibility.test_variable_cell_fire_reference import CellHarmonic, batch


def convergence(fmax=0.01, stress=6.2415e-4):
    return ConvergenceHook(
        criteria=[
            {
                "key": "forces",
                "threshold": fmax,
                "reduce_op": "norm",
                "reduce_dims": -1,
            },
            {
                "key": "stress",
                "threshold": stress,
                "custom_op": lambda x: x.abs().amax((-1, -2)) <= stress,
            },
        ],
        source_status=0,
        target_status=1,
    )


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
@pytest.mark.parametrize(
    "method", ["step", "masked_update", "_masked_pre_update", "_masked_post_update"]
)
def test_inactive_private_state_and_outputs_frozen(cls, method):
    b = batch()
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d.compute(b)
    d.step(b)  # Nonzero cell velocities and adapted control state.
    b.status = torch.tensor([[0], [1]], dtype=torch.long)
    before = {k: v.clone() for k, v in d._state}
    fields = {k: getattr(b, k).clone() for k in d._mutable_fields}
    if method == "step":
        d.step(b)
    else:
        getattr(d, method)(b, torch.tensor([True, False]))
    for key, old in before.items():
        assert torch.equal(getattr(d._state, key)[1], old[1]), key
    for key, old in fields.items():
        start = 2 if key in ("positions", "velocities", "forces") else 1
        assert torch.equal(getattr(b, key)[start:], old[start:]), key


def test_force_and_absolute_stress_are_both_required():
    b = batch()
    b.forces.zero_()
    b.stress.zero_()
    b.stress[0, 0, 0] = -0.1  # A compressive stress must not pass a signed max.
    hook = convergence()
    assert hook.evaluate(b).tolist() == [1]
    b.forces[2, 0] = 0.1
    assert hook.evaluate(b) is None
    b.stress.zero_()
    assert hook.evaluate(b).tolist() == [0]


class Dataset:
    def __init__(self):
        base = batch().to_data_list()
        self.data = [base[0].clone(), base[1].clone(), base[1].clone(), base[0].clone()]
        for i, data in enumerate(self.data):
            data.cell *= 1 + i * 0.03
            data.add_system_property("source_index", torch.tensor([[i]]))

    def __len__(self):
        return len(self.data)

    def get_metadata(self, i):
        return self.data[i].num_nodes, 0

    def __getitem__(self, i):
        return self.data[i].clone(), {}


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_real_sampler_refill_preserves_survivor_and_resets_arrival(cls):
    sampler = SizeAwareSampler(Dataset(), max_atoms=6, max_batch_size=2)
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference", sampler=sampler)
    b = sampler.build_initial_batch()
    d.compute(b)
    d.step(b)
    saved = {k: v[1].clone() for k, v in d._state}
    survivor_id = b.system_id[1].clone()
    b.status = torch.tensor([[1], [0]])
    result = d.refill_check(b, 1)
    assert result.num_graphs == 2
    assert torch.equal(result.system_id[0], survivor_id)
    for key, old in saved.items():
        assert torch.equal(getattr(d._state, key)[0], old), key
    fresh = d._make_new_state(1, result)
    for key, value in fresh:
        assert torch.equal(getattr(d._state, key)[1:], value), key
    assert result.batch_ptr.tolist() == [0, 3, 6]


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_fused_refill_primes_outputs_before_first_arrival_update(cls):
    dataset = Dataset()
    seen = set()

    class CheckPriming:
        stage = DynamicsStage.BEFORE_PRE_UPDATE
        frequency = 1

        def __call__(self, ctx, stage):
            expected = CellHarmonic()(ctx.batch)
            torch.testing.assert_close(ctx.batch.stress, expected["stress"])
            torch.testing.assert_close(ctx.batch.energy.flatten(), expected["energy"])
            seen.update(ctx.batch.source_index.flatten().tolist())

    hook = convergence(stress=1.0)
    d = cls(
        model=CellHarmonic(),
        dt=0.04,
        backend="torch_reference",
        hooks=[hook, CheckPriming()],
        convergence_hook=hook,
    )
    sink = HostMemory(capacity=4)
    stage = FusedStage(
        sub_stages=[(0, d)],
        sampler=SizeAwareSampler(dataset, max_atoms=6, max_batch_size=2),
        sinks=[sink],
        refill_frequency=1,
        device_type="cpu",
    )
    assert stage.run(n_steps=4) is None
    stored = sink.read()
    assert len(sink) == 4
    assert sorted(stored.source_index.flatten().tolist()) == list(range(4))
    assert sorted(stored.system_id.flatten().tolist()) == list(range(4))
    assert seen == set(range(4))
    assert d.step_count == stage.step_count == 2  # Priming is not an update.


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_analytic_joint_convergence_through_native_hooks(cls):
    b = batch()
    initial = CellHarmonic()(b)["energy"]
    hook = convergence(stress=1e-4)
    d = cls(
        model=CellHarmonic(),
        dt=0.04,
        backend="torch_reference",
        hooks=[hook],
        convergence_hook=hook,
    )
    stage = FusedStage(sub_stages=[(0, d)], device_type="cpu")
    result = stage.run(b, n_steps=600)
    assert result.status.flatten().tolist() == [1, 1]
    assert stage.step_count < 600
    assert torch.all(result.energy.flatten() < initial * 1e-4)
    torch.testing.assert_close(result.stress, CellHarmonic()(result)["stress"])


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_vector_parameters_support_batch_but_reject_ambiguous_arrivals(cls):
    b = batch()
    d = cls(
        model=CellHarmonic(), dt=torch.tensor([0.03, 0.04]), backend="torch_reference"
    )
    d.compute(b)
    d.step(b)
    with pytest.raises(ValueError, match="scalar constructor"):
        d._make_new_state(2, b)  # Same cardinality must not silently reuse old params.


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
@pytest.mark.parametrize("fused", [False, True])
def test_hooks_observe_frozen_inactive_geometry_outputs_and_private_state(cls, fused):
    b = batch()
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d.compute(b)
    d.step(b)
    b.status = torch.tensor([[0], [1]])
    old = {k: getattr(b, k).clone() for k in d._mutable_fields}
    state = {k: v[1].clone() for k, v in d._state}
    seen = []

    class Observer:
        frequency = 1

        def __init__(self, stage):
            self.stage = stage

        def __call__(self, ctx, stage):
            seen.append(stage)
            for key, value in old.items():
                start = 2 if key in ("positions", "velocities", "forces") else 1
                assert torch.equal(getattr(ctx.batch, key)[start:], value[start:]), key
            for key, value in state.items():
                assert torch.equal(getattr(d._state, key)[1], value), key

    stages = [DynamicsStage.AFTER_COMPUTE, DynamicsStage.AFTER_POST_UPDATE]
    if not fused:
        stages.append(DynamicsStage.AFTER_PRE_UPDATE)
    for stage in stages:
        d.register_hook(Observer(stage))
    runner = FusedStage(sub_stages=[(0, d)], device_type="cpu") if fused else d
    runner.step(b)
    assert set(seen) == set(stages)


def signature(b):
    return {
        tuple(row)
        for row in torch.cat(
            (b.neighbor_list.long(), b.neighbor_list_shifts.long()), -1
        ).tolist()
    }


def image_oracle(b, cutoff):
    """Independent exhaustive small-cell image enumeration, for tests only."""
    result = set()
    for i, j, shift in product(
        range(b.num_nodes), range(b.num_nodes), product(range(-4, 5), repeat=3)
    ):
        if b.batch_idx[i] != b.batch_idx[j] or (i == j and shift == (0, 0, 0)):
            continue
        h = b.cell[b.batch_idx[i]]
        delta = b.positions[j] - b.positions[i] + torch.tensor(shift, dtype=h.dtype) @ h
        if float(delta.norm()) < cutoff:
            result.add((i, j, *shift))
    return result


@pytest.mark.parametrize("skin", [0.0, 0.4])
def test_current_cell_neighbors_compression_shear_cutoff_images_and_growth(skin):
    data = AtomicData(
        positions=torch.zeros(1, 3, dtype=torch.float64),
        atomic_numbers=torch.ones(1, dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float64)[None] * 3.01,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    b = Batch.from_data_list([data])
    hook = NeighborListHook(
        NeighborConfig(cutoff=3.0),
        skin=skin,
        backend="torch_reference",
        max_neighbors=1,
    )
    ctx = DynamicsContext(batch=b)
    hook(ctx, hook.stage)
    initial_capacity = hook._max_neighbors
    for h in (
        torch.eye(3) * 2.99,
        torch.tensor([[1.1, 0, 0], [0.3, 1.2, 0], [-0.1, 0.2, 1.3]]),
    ):
        b.cell[0].copy_(h)
        hook(ctx, hook.stage)
        assert signature(b) == image_oracle(b, 3.0 + skin)
    assert hook._max_neighbors > initial_capacity
    assert any(abs(entry[2]) >= 2 for entry in signature(b))
    before = signature(b)
    hook(ctx, hook.stage)
    assert signature(b) == before


def test_single_atom_graph_mask_uses_storage_levels_for_cell_and_stress():
    systems = [
        AtomicData(
            **{
                key: value[:1].clone() if key in data.__node_keys__ else value.clone()
                for key, value in data
                if isinstance(value, torch.Tensor)
            }
        )
        for data in batch().to_data_list()
    ]
    b = Batch.from_data_list(systems)
    d = FIREVariableCell(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d.compute(b)
    d.step(b)
    old_h, old_stress = b.cell[1].clone(), b.stress[1].clone()
    b.status = torch.tensor([[0], [1]])
    d.step(b)
    assert torch.equal(b.cell[1], old_h) and torch.equal(b.stress[1], old_stress)
