# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PR194 scientific cases adapted to the retained local Torch reference APIs.

Source: NVIDIA/nvalchemi-toolkit-ops, commit
6967a46d39418cfeb0ea962fa7b85daaaa49a0d0, test/dynamics/test_fire.py and
test/dynamics/test_fire2.py. These tests execute local CPU implementations;
they do not import candidate code or establish native Warp correctness.
"""

import numpy as np
import pytest
import torch

from nvalchemiops.cell_fire_reference import fire2_step_coord_cell
from nvalchemiops.stress_reference import stress_to_cell_force


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keep_aligned", [False, True])
def test_pr194_bond_energy_cell_force_finite_difference(dtype, keep_aligned):
    """A skew row cell distinguishes H^-T sigma from sigma H^-T."""
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    cell = np.array([[3.2, 0, 0], [0.65, 2.8, 0], [-0.4, 0.55, 3.6]], dtype=np_dtype)
    fraction = np.array([0.7, -0.45, 0.3], dtype=np_dtype)
    bond = fraction @ cell
    volume = abs(np.linalg.det(cell))
    stress = np.outer(bond, bond) / volume
    actual = stress_to_cell_force(
        torch.tensor(stress[None], dtype=dtype),
        torch.tensor(cell[None], dtype=dtype),
        torch.tensor([volume], dtype=dtype),
        keep_aligned,
    )[0].numpy()
    epsilon = 2e-3 if dtype == torch.float32 else 1e-6
    atol = 3e-4 if dtype == torch.float32 else 2e-8

    def energy(h):
        r = fraction @ h
        return 0.5 * np.dot(r, r)

    expected = np.zeros_like(cell)
    for row in range(3):
        for column in range(3):
            if keep_aligned and row < column:
                assert actual[row, column] == 0
                continue
            delta = np.zeros_like(cell)
            delta[row, column] = epsilon
            expected[row, column] = -(energy(cell + delta) - energy(cell - delta)) / (2 * epsilon)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pr194_ragged_cell_only_motion_preserves_fractions(dtype):
    """Three unequal systems cover uphill reset and capped affine cell motion."""
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    counts = [3, 6, 5]
    idx = torch.repeat_interleave(torch.arange(3, dtype=torch.int32), torch.tensor(counts))
    positions = torch.tensor(
        (1.5 * np.random.default_rng(525).standard_normal((sum(counts), 3))).astype(np_dtype),
        dtype=dtype,
    )
    cell = torch.tensor([
        [[4.0, 0, 0], [0.65, 4.5, 0], [-0.3, 0.45, 5.0]],
        [[4.4, 0, 0], [-0.55, 4.2, 0], [0.4, 0.3, 4.8]],
        [[3.8, 0, 0], [0.7, 4.6, 0], [-0.5, -0.35, 4.3]],
    ], dtype=dtype)
    cell_force = torch.tensor([
        [[0.4, 0, 0], [0.08, 0.3, 0], [-0.06, 0.04, 0.35]],
        [[0.3, 0, 0], [-0.05, 0.35, 0], [0.04, 0.06, 0.25]],
        [[0.35, 0, 0], [0.07, 0.25, 0], [-0.03, -0.05, 0.4]],
    ], dtype=dtype)
    cell_velocity = torch.stack([-4 * cell_force[0], 4 * cell_force[1], 4 * cell_force[2]])
    before_positions, before_cell = positions.clone(), cell.clone()
    before_fraction = torch.einsum("ni,nij->nj", positions, torch.linalg.inv(cell)[idx.long()])
    alpha = torch.full((3,), 0.09, dtype=dtype)
    dt = torch.full((3,), 0.08, dtype=dtype)
    positive_steps = torch.full((3,), 2, dtype=torch.int32)
    maxstep = 0.003
    fire2_step_coord_cell(
        positions, torch.zeros_like(positions), torch.zeros_like(positions),
        cell, cell_velocity, cell_force, idx, alpha, dt, positive_steps,
        maxstep=maxstep,
    )
    after_fraction = torch.einsum("ni,nij->nj", positions, torch.linalg.inv(cell)[idx.long()])
    tolerance = 4e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(after_fraction, before_fraction, atol=tolerance, rtol=tolerance)
    assert not torch.allclose(cell, before_cell)
    assert torch.equal(positive_steps, torch.tensor([0, 3, 3], dtype=torch.int32))
    assert torch.count_nonzero(cell_velocity[0]) == 0
    assert torch.linalg.vector_norm(cell_velocity[1:]) > 0
    displacement = torch.linalg.vector_norm(positions - before_positions, dim=1)
    assert torch.all(displacement <= maxstep + tolerance)
