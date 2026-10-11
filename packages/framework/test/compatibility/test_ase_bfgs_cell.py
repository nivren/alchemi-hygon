# SPDX-License-Identifier: Apache-2.0
"""ASE UnitCellFilter oracle, continuation and Dynamics lifecycle gates."""

import copy
import os
import subprocess
import sys
from collections import OrderedDict

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import UnitCellFilter
from ase.optimize import BFGS

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from nvalchemi.models.base import ModelConfig
from test.compatibility.test_ase_bfgs import PolynomialModel


class CellModel(PolynomialModel):
    """Rotation invariant E(r,H), exact tensile stress and model call counts."""

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(
            outputs={"energy", "forces", "stress"}, supports_pbc=True
        )

    def forward(self, batch):
        self.calls += 1
        r, h = batch.positions, batch.cell[0]
        k = torch.arange(1, len(r) + 1, dtype=r.dtype, device=r.device)[:, None]
        target = torch.diag(
            torch.tensor([4.0, 3.0, 5.0], dtype=h.dtype, device=h.device)
        )
        diff = h @ h.T - target
        energy = (k * r.square()).sum() / 2 + 0.1 * diff.square().sum() / 2
        f = -k * r
        dh = 0.2 * diff @ h
        stress = (r.T @ (-f) + h.T @ dh) / torch.linalg.det(h).abs()
        return OrderedDict(energy=energy.reshape(1, 1), forces=f, stress=stress[None])


class CellCalculator(Calculator):
    """Independent NumPy potential for the ASE optimizer."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        r, h = self.atoms.positions, self.atoms.cell.array
        k = np.arange(1, len(r) + 1)[:, None]
        diff = h @ h.T - np.diag([4.0, 3.0, 5.0])
        e = (k * r**2).sum() / 2 + 0.1 * (diff**2).sum() / 2
        f = -k * r
        dh = 0.2 * diff @ h
        stress = (r.T @ (-f) + h.T @ dh) / abs(np.linalg.det(h))
        self.results = dict(energy=e, free_energy=e, forces=f, stress=stress)


def make_batch(device="cpu"):
    r = torch.tensor([[0.2, -0.1, 0.4], [0.6, 0.3, -0.2]], dtype=torch.float64)
    h = torch.tensor([[2.2, 0.3, 0.1], [0.2, 1.8, 0.4], [0.1, 0.2, 2.3]], dtype=r.dtype)
    b = Batch.from_data_list(
        [
            AtomicData(
                positions=r,
                atomic_numbers=torch.ones(2, dtype=torch.long),
                cell=h[None],
                pbc=torch.ones(1, 3, dtype=torch.bool),
                forces=torch.zeros_like(r),
                energy=torch.zeros(1, 1, dtype=r.dtype),
                stress=torch.zeros(1, 3, 3, dtype=r.dtype),
            )
        ]
    ).to(device)
    b.system_id = torch.tensor([[17]], device=device)
    return b


def assert_snapshot_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], dict):
            assert_snapshot_equal(a[key], b[key])
        elif isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"cell_factor": 3.1, "scalar_pressure": 0.037},
        {"mask": [1, 0, 1, 0, 1, 0]},
        {"hydrostatic_strain": True},
        {"constant_volume": True},
        {"mask": [0, 1, 1, 0, 0, 1], "constant_volume": True},
        {"hydrostatic_strain": True, "constant_volume": True},
        {"orig_cell": [[2.0, 0.1, 0.3], [0.1, 1.9, 0.2], [0.2, 0.1, 2.1]]},
    ],
)
def test_step_matches_ase_full_chart_hessian_history_and_geometry(options):
    b = make_batch()
    atoms = Atoms("H2", positions=b.positions.numpy(), cell=b.cell[0].numpy(), pbc=True)
    atoms.calc = CellCalculator()
    filter_ = UnitCellFilter(atoms, **options)
    ase = BFGS(filter_, logfile=None, alpha=53, maxstep=0.12)
    d = ASECompatibleBFGSVariableCell(
        CellModel(), alpha=53, maxstep=0.12, fmax=1e-10, **options
    )
    for i in range(8):
        before = filter_.get_positions()
        force = filter_.get_forces()
        ase.step()
        d.step(b)
        np.testing.assert_allclose(
            b.positions.numpy(), atoms.positions, atol=3e-12, rtol=3e-12
        )
        np.testing.assert_allclose(
            b.cell[0].numpy(), atoms.cell.array, atol=3e-12, rtol=3e-12
        )
        np.testing.assert_allclose(
            d.bfgs_state.hessian.numpy(), ase.H, atol=5e-10, rtol=3e-11
        )
        np.testing.assert_allclose(
            d.bfgs_state.previous_positions.numpy(), before, atol=3e-12
        )
        np.testing.assert_allclose(
            d.bfgs_state.previous_forces.numpy(), force, atol=3e-12
        )
        np.testing.assert_allclose(
            d.filter_state.pack(b.positions, b.cell[0]).numpy(),
            filter_.get_positions(),
            atol=3e-12,
        )
        np.testing.assert_allclose(
            d._generalized_forces(b).numpy(), filter_.get_forces(), atol=3e-12
        )
        assert d.model.calls == i + 2
        assert d.bfgs_state.step_count == d.step_count == i + 1
        assert d.last_step.applied_step.norm(dim=1).max() <= 0.12 + 1e-15
    assert d.bfgs_state.num_scalar_dofs == 15


def test_joint_convergence_matches_ase_and_preserves_physical_energy():
    b = make_batch()
    atoms = Atoms("H2", positions=b.positions.numpy(), cell=b.cell[0].numpy(), pbc=True)
    atoms.calc = CellCalculator()
    oracle = BFGS(UnitCellFilter(atoms, scalar_pressure=0.02), logfile=None)
    oracle.run(fmax=0.05, steps=100)
    d = ASECompatibleBFGSVariableCell(CellModel(), scalar_pressure=0.02)
    d.run(b, n_steps=100)
    assert d.bfgs_state.step_count == d.step_count == oracle.nsteps
    assert d.model.calls == d.step_count + 1
    assert d._generalized_forces(b).norm(dim=1).max() < 0.05
    np.testing.assert_allclose(b.cell[0].numpy(), atoms.cell.array, atol=3e-10)
    torch.testing.assert_close(
        d.get_enthalpy(b),
        b.energy + 0.02 * torch.linalg.det(b.cell).abs().reshape(1, 1),
    )


@pytest.mark.parametrize("mask,updates", [(None, 1), ([0] * 6, 0)])
def test_initial_atomic_zero_force_with_stress_uses_joint_criterion(mask, updates):
    b = make_batch()
    b.positions.zero_()
    d = ASECompatibleBFGSVariableCell(CellModel(), mask=mask)
    d.run(b, n_steps=1)
    assert d.bfgs_state.step_count == updates


@pytest.mark.parametrize(
    "method", ["step", "masked_update", "_masked_pre_update", "_masked_post_update"]
)
def test_inactive_entrances_freeze_geometry_filter_and_dense_history(method):
    b = make_batch()
    d = ASECompatibleBFGSVariableCell(CellModel(), fmax=1e-10)
    d.run(b, n_steps=2)
    saved, old = d.state_dict(), b.clone()
    if method == "step":
        b.status = torch.tensor([[d.exit_status]])
        d.step(b)
    else:
        getattr(d, method)(b, torch.tensor([False]))
    # Base scheduling visits can increment on an explicitly inactive step.
    now = d.state_dict()
    now["step_count"] = saved["step_count"]
    assert_snapshot_equal(saved, now)
    for key in ("positions", "cell", "forces", "stress", "energy"):
        assert torch.equal(getattr(b, key), getattr(old, key))


def test_restart_preserves_reference_cell_parameters_and_trajectory():
    b = make_batch()
    d = ASECompatibleBFGSVariableCell(
        CellModel(), scalar_pressure=0.02, cell_factor=2.7, fmax=1e-10
    )
    d.run(b, n_steps=4)
    saved, geometry = d.state_dict(), b.clone()
    d.run(b, n_steps=4)
    resumed = ASECompatibleBFGSVariableCell(CellModel())
    resumed.load_state_dict(saved, device="cpu")
    geometry.forces.fill_(999)
    geometry.stress.fill_(999)
    resumed.run(geometry, n_steps=4)
    torch.testing.assert_close(geometry.positions, b.positions, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(geometry.cell, b.cell, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(
        resumed.bfgs_state.hessian, d.bfgs_state.hessian, atol=3e-12, rtol=3e-12
    )
    assert resumed.model.calls == 5
    assert_snapshot_equal(
        resumed.filter_state.state_dict(), d.filter_state.state_dict()
    )


@pytest.mark.parametrize(
    "target", ["positions", "cell", "atomic_numbers", "pbc", "system_id"]
)
def test_restart_rejects_mismatched_paired_batch_before_model_evaluation(target):
    b = make_batch()
    d = ASECompatibleBFGSVariableCell(CellModel())
    d.run(b, n_steps=1)
    resumed = ASECompatibleBFGSVariableCell(CellModel())
    resumed.load_state_dict(d.state_dict())
    if target == "pbc":
        b.pbc.logical_not_()
    else:
        getattr(b, target).view(-1)[0] += 1
    with pytest.raises(ValueError):
        resumed.step(b)
    assert resumed.model.calls == 0


@pytest.mark.parametrize("target", ["positions", "cell", "stress", "forces"])
def test_external_geometry_or_output_mutation_invalidates_cache(target):
    b = make_batch()
    d = ASECompatibleBFGSVariableCell(CellModel(), fmax=1e-10)
    d.run(b, n_steps=1)
    getattr(b, target).view(-1)[0] += 0.001
    d.step(b)
    assert d.model.calls == 4


@pytest.mark.parametrize(
    "bad", ["cell", "positions", "filter", "mode", "count", "fmax", "signature", "kind"]
)
def test_bad_snapshot_load_is_atomic(bad):
    b = make_batch()
    d = ASECompatibleBFGSVariableCell(CellModel())
    d.run(b, n_steps=1)
    saved = d.state_dict()
    invalid = copy.deepcopy(saved)
    if bad in ("positions", "cell"):
        invalid[bad].fill_(float("nan"))
    elif bad == "filter":
        invalid["filter_state"]["orig_cell"].zero_()
    elif bad == "mode":
        invalid["bfgs_state"]["coordinate_mode"] = "cartesian"
    elif bad == "count":
        invalid["step_count"] = -1
    elif bad == "fmax":
        invalid["fmax"] = 0
    elif bad == "signature":
        invalid["signature"]["pbc"] = torch.zeros(1, 3)
    else:
        invalid["kind"] = "fixed"
    with pytest.raises((ValueError, TypeError)):
        d.load_state_dict(invalid)
    assert_snapshot_equal(saved, d.state_dict())


def test_singular_candidate_rolls_back_geometry_and_bfgs_history(monkeypatch):
    from nvalchemi.dynamics.optimizers import bfgs_ase_cell as module

    b = make_batch()
    d = ASECompatibleBFGSVariableCell(CellModel())
    d._ensure_state_initialized(b)
    d._prime(b)
    d._checkpoint_positions = b.positions.clone()
    before, old = d.state_dict(), b.clone()
    original = module.bfgs_ase_step

    def singular(q, *args, **kwargs):
        diagnostics = original(q, *args, **kwargs)
        q[-3:].zero_()
        return diagnostics

    monkeypatch.setattr(module, "bfgs_ase_step", singular)
    with pytest.raises(ValueError, match="singular"):
        d.pre_update(b)
    assert_snapshot_equal(before, d.state_dict())
    assert torch.equal(b.positions, old.positions)
    assert torch.equal(b.cell, old.cell)


@pytest.mark.parametrize("slot", ["cell", "stress"])
def test_alias_slots_rejected_before_model_changes_data(slot):
    b = make_batch()
    setattr(b, slot, torch.zeros(9, dtype=torch.float64).reshape(1, 3, 3))
    other = "stress" if slot == "cell" else "cell"
    setattr(b, other, getattr(b, slot))
    d = ASECompatibleBFGSVariableCell(CellModel())
    with pytest.raises(ValueError, match="alias"):
        d.step(b)
    assert d.model.calls == 0


def test_mode_selection_resolves_once_and_lazy_public_imports(monkeypatch):
    from nvalchemi.dynamics.optimizers import bfgs_ase_cell as module

    calls = []
    original = module.resolve_compute_backend

    def resolve(*args, **kwargs):
        calls.append(kwargs["features"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "resolve_compute_backend", resolve)
    d = ASECompatibleBFGSVariableCell(CellModel())
    d.run(make_batch(), n_steps=3)
    assert calls == [{"single_system", "variable_cell", "ase_unitcell"}]
    code = "from nvalchemi.dynamics import ASECompatibleBFGSVariableCell; import sys; assert 'warp' not in sys.modules; assert 'nvalchemiops.bfgs_ase_reference' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)  # noqa: S603 - import-only code


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "gradient", "layout", "singular", "batch"]
)
def test_invalid_geometry_admission_precedes_model_evaluation(invalid):
    b = make_batch()
    if invalid == "shape":
        b.positions = b.positions.unsqueeze(-1)
    elif invalid == "dtype":
        b.stress = b.stress.float()
    elif invalid == "gradient":
        b.cell.requires_grad_()
    elif invalid == "layout":
        b.stress = b.stress.transpose(-1, -2)
    elif invalid == "singular":
        b.cell.zero_()
    else:
        b = Batch.from_data_list(b.to_data_list() * 2)
    d = ASECompatibleBFGSVariableCell(CellModel())
    with pytest.raises(ValueError):
        d.step(b)
    assert d.model.calls == 0
