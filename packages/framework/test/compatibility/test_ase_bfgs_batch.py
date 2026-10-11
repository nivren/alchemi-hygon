# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous BFGS, native FusedStage, identity/refill and restart gates."""

import copy
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.filters import UnitCellFilter
from ase.optimize import BFGS

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    FIRE2,
    ASECompatibleBFGS,
    ASECompatibleBFGSVariableCell,
    FusedStage,
)
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from test.compatibility.test_ase_bfgs import (
    PolynomialCalculator,
    PolynomialModel,
    make_batch,
)
from test.compatibility.test_ase_bfgs_cell import (
    CellCalculator,
    CellModel,
    assert_snapshot_equal,
)
from test.compatibility.test_ase_bfgs_cell import (
    make_batch as cell_batch,
)


class BatchPolynomial(PolynomialModel):
    def forward(self, batch):
        self.calls += 1
        energies, forces = [], []
        for a, b in zip(batch.batch_ptr.tolist(), batch.batch_ptr.tolist()[1:]):
            q = batch.positions[a:b]
            k = torch.arange(1, q.numel() + 1, dtype=q.dtype, device=q.device).view_as(
                q
            )
            energies.append((k * q * q / 2 + 0.3 * q**4 / 4).sum())
            forces.append(-(k * q + 0.3 * q**3))
        return OrderedDict(
            energy=torch.stack(energies)[:, None], forces=torch.cat(forces)
        )


class BatchCell(CellModel):
    def forward(self, batch):
        self.calls += 1
        outputs = []
        ptr = batch.batch_ptr.tolist()
        for i, (a, b) in enumerate(zip(ptr, ptr[1:])):
            r, h = batch.positions[a:b], batch.cell[i]
            k = torch.arange(1, len(r) + 1, dtype=r.dtype, device=r.device)[:, None]
            diff = h @ h.T - torch.diag(
                torch.tensor([4.0, 3.0, 5.0], dtype=r.dtype, device=r.device)
            )
            e = (k * r * r).sum() / 2 + 0.1 * diff.square().sum() / 2
            f = -k * r
            dh = 0.2 * diff @ h
            stress = (r.T @ (-f) + h.T @ dh) / torch.linalg.det(h).abs()
            outputs.append((e, f, stress))
        return OrderedDict(
            energy=torch.stack([o[0] for o in outputs])[:, None],
            forces=torch.cat([o[1] for o in outputs]),
            stress=torch.stack([o[2] for o in outputs]),
        )


def mixed(variable=False, device="cpu"):
    data = []
    for i, n in enumerate((2, 3, 2)):
        if variable:
            single = cell_batch()
            row = single.to_data_list()[0]
            if n == 3:
                q = torch.cat(
                    (
                        row.positions,
                        torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64),
                    )
                )
                row = AtomicData(
                    positions=q,
                    atomic_numbers=torch.ones(3, dtype=torch.long),
                    forces=torch.zeros_like(q),
                    energy=row.energy.clone(),
                    cell=row.cell.clone(),
                    stress=row.stress.clone(),
                    pbc=row.pbc.clone(),
                )
        else:
            row = make_batch(n).to_data_list()[0]
        row.positions *= 1 + i * 0.03
        row.system_id = torch.tensor([[11 + i]])
        data.append(row)
    return Batch.from_data_list(data).to(device)


@pytest.mark.parametrize(
    "variable,options",
    [
        (False, {}),
        (True, {}),
        (True, dict(cell_factor=2.7, scalar_pressure=0.037)),
        (True, dict(mask=[1, 0, 1, 0, 1, 0])),
    ],
)
def test_heterogeneous_steps_match_ase_and_independent_runs(variable, options):
    b = mixed(variable)
    model = BatchCell() if variable else BatchPolynomial()
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    dynamics = cls(model, alpha=53, maxstep=0.12, fmax=1e-10, **options)
    independent = []
    oracles = []
    for row in b.to_data_list():
        single = Batch.from_data_list([row.clone()])
        independent.append(
            (
                single,
                cls(
                    CellModel() if variable else PolynomialModel(0.3),
                    alpha=53,
                    maxstep=0.12,
                    fmax=1e-10,
                    **options,
                ),
            )
        )
        atoms = Atoms(
            numbers=row.atomic_numbers.numpy(),
            positions=row.positions.numpy(),
            cell=row.cell[0].numpy(),
            pbc=row.pbc[0].numpy(),
        )
        atoms.calc = CellCalculator() if variable else PolynomialCalculator(0.3)
        target = UnitCellFilter(atoms, **options) if variable else atoms
        oracles.append(BFGS(target, logfile=None, alpha=53, maxstep=0.12))
    for count in range(7):
        dynamics.step(b)
        for i, ((single, d), oracle) in enumerate(zip(independent, oracles)):
            d.step(single)
            oracle.step()
            a, z = b.batch_ptr.tolist()[i : i + 2]
            np.testing.assert_allclose(
                b.positions[a:z].numpy(), oracle.atoms.positions, atol=4e-12, rtol=4e-12
            )
            torch.testing.assert_close(
                b.positions[a:z], single.positions, atol=4e-12, rtol=4e-12
            )
            state = dynamics.bfgs_states[11 + i]
            np.testing.assert_allclose(
                state.hessian.numpy(), oracle.H, atol=6e-10, rtol=6e-11
            )
            assert state.step_count == count + 1
            if variable:
                np.testing.assert_allclose(
                    b.cell[i].numpy(),
                    oracle.atoms.atoms.cell.array,
                    atol=4e-12,
                    rtol=4e-12,
                )
                assert dynamics.filter_states[11 + i].cell_factor == options.get(
                    "cell_factor", z - a
                )
        assert model.calls == count + 2


@pytest.mark.parametrize("variable", [False, True])
def test_reorder_exit_arrival_and_restart_preserve_ids(variable):
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    model_cls = BatchCell if variable else BatchPolynomial
    b = mixed(variable)
    d = cls(model_cls(), fmax=1e-10)
    for _ in range(3):
        d.step(b)
    old = {i: s.state_dict() for i, s in d.bfgs_states.items()}
    reordered = b.index_select(torch.tensor([2, 0, 1]))
    d._ensure_state_initialized(reordered)
    for i, s in d.bfgs_states.items():
        assert_snapshot_equal(old[i], s.state_dict())
    snapshot = d.state_dict()
    resumed = cls(model_cls(), fmax=1e-10)
    resumed.load_state_dict(snapshot)
    paired = reordered.clone()
    d.step(reordered)
    resumed.step(paired)
    torch.testing.assert_close(reordered.positions, paired.positions, atol=0, rtol=0)
    if variable:
        torch.testing.assert_close(reordered.cell, paired.cell, atol=0, rtol=0)
    for i in d.bfgs_states:
        assert_snapshot_equal(
            d.bfgs_states[i].state_dict(), resumed.bfgs_states[i].state_dict()
        )
    rows = reordered.to_data_list()
    rows[2].system_id.fill_(90)
    replaced = Batch.from_data_list(rows[:1] + rows[2:])
    survivor = d.bfgs_states[13]
    d._ensure_state_initialized(replaced)
    assert set(d.bfgs_states) == {13, 90} and d.bfgs_states[13] is survivor
    assert d.bfgs_states[90].step_count == 0 and not d.bfgs_states[90].initialized


@pytest.mark.parametrize("variable", [False, True])
def test_snapshot_validation_failure_is_atomic(variable):
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    d = cls(BatchCell() if variable else BatchPolynomial())
    d.step(mixed(variable))
    before = d.state_dict()
    bad = copy.deepcopy(before)
    bad["entries"][13]["snapshot"]["bfgs_state"]["step_count"] = -1
    with pytest.raises(ValueError):
        d.load_state_dict(bad)
    assert_snapshot_equal(before, d.state_dict())


def test_initial_convergence_and_inactive_state_freeze():
    b = mixed()
    b.positions[:2].zero_()
    b.status = torch.tensor([[0], [0], [1]])
    d = ASECompatibleBFGS(BatchPolynomial(), fmax=0.05)
    original = b.positions.clone()
    for _ in range(4):
        d.step(b)
    assert d.bfgs_states[11].step_count == d.bfgs_states[13].step_count == 0
    assert d.bfgs_states[12].step_count == 4
    assert torch.equal(b.positions[:2], original[:2]) and torch.equal(
        b.positions[5:], original[5:]
    )


class Dataset:
    def __init__(self, variable=False):
        self.rows = mixed(variable).to_data_list()
        self.rows += [self.rows[0].clone()]
        for i, row in enumerate(self.rows):
            row.add_system_property("source_index", torch.tensor([[i]]))
            row.add_node_property("velocities", torch.zeros_like(row.positions))

    def __len__(self):
        return len(self.rows)

    def get_metadata(self, i):
        return self.rows[i].num_nodes, 0

    def __getitem__(self, i):
        return self.rows[i].clone(), {}


@pytest.mark.parametrize("variable", [False, True])
def test_native_sampler_refill_preserves_survivor_and_initializes_arrival(variable):
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    sampler = SizeAwareSampler(Dataset(variable), max_atoms=6, max_batch_size=2)
    d = cls(BatchCell() if variable else BatchPolynomial(), fmax=1e-10, sampler=sampler)
    b = sampler.build_initial_batch()
    d.step(b)
    survivor_id = int(b.system_id[1])
    survivor = d.bfgs_states[survivor_id]
    saved = survivor.state_dict()
    b.status = torch.tensor([[1], [0]])
    result = d.refill_check(b, 1)
    assert d.bfgs_states[survivor_id] is survivor
    assert_snapshot_equal(saved, survivor.state_dict())
    arrival = int(result.system_id[1])
    assert arrival != survivor_id and d.bfgs_states[arrival].step_count == 0
    expected = d.model(result)
    torch.testing.assert_close(result.forces, expected["forces"])


@pytest.mark.parametrize("variable", [False, True])
def test_fused_shared_model_convergence_and_budgeted_refill(variable):
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    model = BatchCell() if variable else BatchPolynomial()
    d = cls(model, fmax=1e-10, n_steps=3)
    sink = HostMemory(capacity=4)
    stage = FusedStage(
        [(0, d)],
        sampler=SizeAwareSampler(Dataset(variable), max_atoms=6, max_batch_size=2),
        sinks=[sink],
        refill_frequency=1,
        device_type="cpu",
    )
    assert stage.run(n_steps=20) is None
    assert len(sink) == 4 and sorted(
        sink.read().source_index.flatten().tolist()
    ) == list(range(4))
    assert model.calls == stage.step_count + 2  # initial + one admission priming
    assert not d.bfgs_states


def test_fire2_to_bfgs_shared_compute_preserves_stage_history():
    b = mixed()
    b.velocities = torch.zeros_like(b.positions)
    model = BatchPolynomial()
    fire = FIRE2(model=model, dt=0.01, n_steps=2, backend="torch_reference")
    bfgs = ASECompatibleBFGS(model, fmax=1e-10, n_steps=3)
    stage = FusedStage([(0, fire), (1, bfgs)], device_type="cpu")
    stage.run(b, n_steps=5)
    # Native counters count the migration iteration towards the next stage's
    # budget. Numerical history counts only actual BFGS updates.
    assert model.calls == stage.step_count + 1 == 5
    assert all(s.step_count == 2 for s in bfgs.bfgs_states.values())
    assert b.status.flatten().tolist() == [2, 2, 2]


def test_restored_filter_options_apply_to_new_arrivals():
    b = mixed(True)
    original = ASECompatibleBFGSVariableCell(
        BatchCell(), cell_factor=2.7, scalar_pressure=0.037, mask=[1, 0, 1, 0, 1, 0]
    )
    original.step(b)
    resumed = ASECompatibleBFGSVariableCell(BatchCell())
    resumed.load_state_dict(original.state_dict())
    resumed._ensure_state_initialized(b)
    rows = b.to_data_list()
    rows[2].system_id.fill_(90)
    resumed._ensure_state_initialized(Batch.from_data_list(rows))
    adapter = resumed.filter_states[90]
    assert adapter.cell_factor == 2.7 and adapter.scalar_pressure == 0.037
    assert torch.equal(adapter.mask, resumed.filter_states[11].mask)


def test_batch_restart_rejects_wrong_membership_and_alias():
    b = mixed()
    d = ASECompatibleBFGS(BatchPolynomial())
    d.step(b)
    resumed = ASECompatibleBFGS(BatchPolynomial())
    resumed.load_state_dict(d.state_dict())
    wrong = b.clone()
    wrong.system_id[2] = 90
    with pytest.raises(ValueError, match="paired system IDs"):
        resumed.step(wrong)
    d.bfgs_states[11].previous_positions = b.positions[:2]
    with pytest.raises(ValueError, match="alias"):
        d.step(b)


def test_single_system_fused_uses_shared_model_only():
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel(0.3), fmax=1e-10)
    stage = FusedStage([(0, d)], device_type="cpu")
    stage.run(b, n_steps=3)
    assert d.model.calls == 4 and d.bfgs_state.step_count == 3


def test_batch_selection_is_cached_through_reorder():
    from nvalchemi.dynamics.optimizers.bfgs_ase_batch import resolve_compute_backend

    b = mixed(True)
    d = ASECompatibleBFGSVariableCell(BatchCell(), fmax=1e-10)
    with patch(
        "nvalchemi.dynamics.optimizers.bfgs_ase_batch.resolve_compute_backend",
        wraps=resolve_compute_backend,
    ) as resolve:
        d.step(b)
        d.step(b.index_select(torch.tensor([2, 0, 1])))
    assert resolve.call_count == 1
    assert resolve.call_args.kwargs["features"] == {
        "single_system",
        "batch",
        "heterogeneous",
        "variable_cell",
        "ase_unitcell",
    }


def test_variable_cell_convergence_freezes_only_eligible_system():
    b = mixed(True)
    b.positions[:2].zero_()
    original_cell = b.cell.clone()
    d = ASECompatibleBFGSVariableCell(BatchCell(), mask=[0] * 6, fmax=0.05)
    d.run(b, n_steps=4)
    assert d.bfgs_states[11].step_count == 0 and d.bfgs_states[12].step_count == 4
    assert torch.equal(b.cell, original_cell)
    assert torch.equal(b.positions[:2], torch.zeros_like(b.positions[:2]))


def check_single_restart_promotion(variable, device="cpu", backend="torch_reference"):
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    model_cls = BatchCell if variable else BatchPolynomial
    batch = mixed(variable)
    single = batch.index_select(torch.tensor([0]))
    original = cls(model_cls(), fmax=1e-10)
    original.run(single, n_steps=2)
    saved = original.state_dict()
    paired = Batch.from_data_list(
        [single.to_data_list()[0], batch.to_data_list()[1]]
    ).to(device)
    resumed = cls(model_cls(), backend=backend)
    resumed.load_state_dict(saved, device=device)
    missing = paired.index_select(torch.tensor([1], device=device))
    missing.system_id.fill_(99)
    # Force the Batch adapter for the missing-ID check without a sampler.
    missing = Batch.from_data_list(
        missing.to_data_list() + [batch.to_data_list()[2].to(device)]
    )
    with pytest.raises(ValueError, match="paired system ID"):
        resumed._ensure_state_initialized(missing)
    mutations = ["positions"] + (["cell"] if variable else [])
    for key in mutations:
        wrong = paired.clone()
        getattr(wrong, key)[0] += 0.01
        resumed.load_state_dict(saved, device=device)
        with pytest.raises(ValueError, match="paired physical geometry"):
            resumed._ensure_state_initialized(wrong)
    resumed.load_state_dict(saved, device=device)
    resumed._ensure_state_initialized(paired)
    assert resumed.bfgs_states[11].step_count == 2
    assert resumed.bfgs_states[12].step_count == 0
    original.step(single)
    resumed.step(paired)
    torch.testing.assert_close(
        paired.positions[:2].cpu(), single.positions, atol=5e-12, rtol=5e-12
    )
    torch.testing.assert_close(
        resumed.bfgs_states[11].hessian.cpu(),
        original.bfgs_state.hessian,
        atol=5e-11,
        rtol=5e-12,
    )
    if variable:
        torch.testing.assert_close(
            paired.cell[:1].cpu(), single.cell, atol=5e-12, rtol=5e-12
        )


@pytest.mark.parametrize("variable", [False, True])
def test_single_restart_promotes_with_geometry_guard_and_fresh_arrival(variable):
    check_single_restart_promotion(variable)
