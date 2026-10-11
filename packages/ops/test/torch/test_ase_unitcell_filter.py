# SPDX-License-Identifier: Apache-2.0
"""Independent ASE 3.29 oracle for the complete UnitCellFilter chart."""

import copy

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import UnitCellFilter

from nvalchemiops.ase_unitcell_filter import ASEUnitCellFilter
from nvalchemiops.bfgs_ase import ASEBFGSState, bfgs_ase_prepare_state


def geometry(left_handed=False):
    h0 = torch.tensor(
        [[2.1, 0.3, 0.1], [0.2, 1.8, 0.4], [0.1, 0.2, 2.3]], dtype=torch.float64
    )
    if left_handed:
        h0[0].neg_()
    h = h0 @ torch.tensor(
        [[1.08, 0.03, 0.01], [0.02, 0.96, 0.04], [-0.01, 0.01, 1.03]], dtype=h0.dtype
    )
    r = torch.tensor([[0.2, -0.1, 0.4], [0.6, 0.3, -0.2]], dtype=h.dtype)
    return r, h.contiguous(), h0


@pytest.mark.parametrize("left_handed", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"cell_factor": 3.2, "scalar_pressure": 0.02},
        {"cell_factor": -3.2, "scalar_pressure": 0.02},
        {"mask": [1, 0, 1, 0, 1, 0]},
        {"mask": [[1, 0.2, 0], [0.1, 1, 0], [0, 0, 0.5]]},
        {"hydrostatic_strain": True},
        {"constant_volume": True},
        {"hydrostatic_strain": True, "constant_volume": True, "scalar_pressure": -0.03},
        {"mask": [0, 1, 1, 0, 0, 1], "constant_volume": True},
    ],
)
def test_pack_unpack_forces_enthalpy_against_ase(options, left_handed):
    r, h, h0 = geometry(left_handed)
    f = torch.arange(6, dtype=r.dtype).reshape(2, 3) / 7
    stress = torch.tensor(
        [[0.2, -0.1, 0.03], [-0.1, 0.4, 0.07], [0.03, 0.07, -0.3]], dtype=r.dtype
    )
    atoms = Atoms("H2", positions=r.numpy(), cell=h.numpy(), pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=1.2, free_energy=1.2, forces=f.numpy(), stress=stress.numpy()
    )
    oracle = UnitCellFilter(atoms, orig_cell=h0.numpy(), **options)
    local = ASEUnitCellFilter.prepare(h, 2, orig_cell=h0, **options)
    q = local.pack(r, h)
    np.testing.assert_allclose(q.numpy(), oracle.get_positions(), atol=1e-14)
    gf, projected = local.generalized_forces(f, stress, h)
    np.testing.assert_allclose(gf.numpy(), oracle.get_forces(), atol=1e-14)
    np.testing.assert_allclose(projected.numpy(), oracle.stress, atol=1e-14)
    assert local.enthalpy(torch.tensor(1.2, dtype=r.dtype), h).item() == pytest.approx(
        oracle.get_potential_energy()
    )
    moved = q + torch.linspace(-0.02, 0.03, q.numel(), dtype=r.dtype).reshape_as(q)
    new_r, new_h = local.unpack(moved)
    oracle.set_positions(moved.numpy())
    np.testing.assert_allclose(new_r.numpy(), atoms.positions, atol=1e-14)
    np.testing.assert_allclose(new_h.numpy(), atoms.cell.array, atol=1e-14)
    assert torch.equal(h0, geometry(left_handed)[2])


def energy_forces_stress(r, h):
    """Rotation invariant potential, evaluated by Torch only in this test."""
    target = torch.diag(torch.tensor([4.0, 3.0, 5.0], dtype=h.dtype, device=h.device))
    k = torch.arange(1, r.shape[0] + 1, dtype=r.dtype, device=r.device)[:, None]
    diff = h @ h.T - target
    energy = (k * r.square()).sum() / 2 + 0.1 * diff.square().sum() / 2
    f = -k * r
    dh = 0.2 * diff @ h
    stress = (r.T @ (-f) + h.T @ dh) / torch.linalg.det(h).abs()
    return energy, f, stress.contiguous()


def test_all_3n_plus_9_generalized_forces_finite_difference_with_pressure():
    r, h, h0 = geometry()
    adapter = ASEUnitCellFilter.prepare(
        h, 2, orig_cell=h0, scalar_pressure=0.037, cell_factor=2.7
    )
    q = adapter.pack(r, h)
    e, f, stress = energy_forces_stress(r, h)
    gf, _ = adapter.generalized_forces(f, stress, h)
    finite = torch.empty_like(q)
    eps = 1e-6
    for j in range(q.numel()):
        plus, minus = q.clone(), q.clone()
        plus.view(-1)[j] += eps
        minus.view(-1)[j] -= eps
        rp, hp = adapter.unpack(plus)
        rm, hm = adapter.unpack(minus)
        ep = adapter.enthalpy(energy_forces_stress(rp, hp)[0], hp)
        em = adapter.enthalpy(energy_forces_stress(rm, hm)[0], hm)
        finite.view(-1)[j] = -(ep - em) / (2 * eps)
    torch.testing.assert_close(gf, finite, atol=4e-9, rtol=4e-8)


def test_filter_snapshot_and_generalized_bfgs_state_are_independent():
    r, h, h0 = geometry()
    adapter = ASEUnitCellFilter.prepare(h, 2, orig_cell=h0, mask=[1, 0, 1, 0, 1, 0])
    saved = adapter.state_dict()
    restored = ASEUnitCellFilter.from_state_dict(saved)
    saved["orig_cell"].zero_()
    torch.testing.assert_close(restored.pack(r, h), adapter.pack(r, h))
    state = bfgs_ase_prepare_state(15, coordinate_mode="ase_unitcell")
    assert state.coordinate_mode == "ase_unitcell"
    assert state.state_dict()["schema_version"] == 2
    assert (
        ASEBFGSState.from_state_dict(state.state_dict()).coordinate_mode
        == "ase_unitcell"
    )
    assert bfgs_ase_prepare_state(6).state_dict()["schema_version"] == 1
    assert "coordinate_mode" not in bfgs_ase_prepare_state(6).state_dict()
    with pytest.raises(ValueError):
        bfgs_ase_prepare_state(9, coordinate_mode="ase_unitcell")
    with pytest.raises(ValueError):
        bfgs_ase_prepare_state(15, coordinate_mode="unknown")
    bad = copy.deepcopy(restored.state_dict())
    bad["orig_cell"].zero_()
    with pytest.raises(ValueError):
        ASEUnitCellFilter.from_state_dict(bad)


@pytest.mark.parametrize(
    "options",
    [
        {"mask": [1, 1]},
        {"cell_factor": 0},
        {"cell_factor": float("nan")},
        {"scalar_pressure": float("inf")},
        {"hydrostatic_strain": "yes"},
        {"constant_volume": 1},
        {"orig_cell": torch.zeros(3, 3, dtype=torch.float64)},
    ],
)
def test_invalid_filter_parameters(options):
    _, h, _ = geometry()
    with pytest.raises((ValueError, TypeError)):
        ASEUnitCellFilter.prepare(h, 2, **options)


def test_geometry_failures_are_explicit_and_do_not_mutate_inputs():
    r, h, _ = geometry()
    adapter = ASEUnitCellFilter.prepare(h, 2)
    saved = h.clone()
    q = adapter.pack(r, h)
    q[-3:].zero_()
    with pytest.raises(ValueError, match="singular"):
        adapter.unpack(q)
    assert torch.equal(h, saved)
    with pytest.raises(ValueError):
        adapter.pack(r.float(), h)
    with pytest.raises(ValueError):
        adapter.pack(r.requires_grad_(), h)
