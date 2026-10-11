# SPDX-License-Identifier: Apache-2.0
"""Batched BFGS oracle and all-group commit gates."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS

from nvalchemiops.bfgs_ase import (
    ASEBFGSState,
    BFGSNumericalError,
    bfgs_ase_batch_step,
    bfgs_ase_prepare_state,
    bfgs_ase_step,
)


class Polynomial(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        q = self.atoms.positions
        k = np.arange(1, q.size + 1).reshape(q.shape)
        self.results = {
            "energy": (k * q * q / 2 + 0.3 * q**4 / 4).sum(),
            "forces": -(k * q + 0.3 * q**3),
        }


def setup(mode="cartesian"):
    qs = [
        torch.linspace(-0.4, 0.6 + i * 0.01, n * 3, dtype=torch.float64).reshape(n, 3)
        for i, n in enumerate((2, 3, 2))
    ]
    states = [
        bfgs_ase_prepare_state(
            q.numel(), alpha=53.0, maxstep=0.12, system_id=i, coordinate_mode=mode
        )
        for i, q in enumerate(qs)
    ]
    return qs, states


def forces(qs):
    return [
        -(torch.arange(1, q.numel() + 1, dtype=q.dtype).reshape_as(q) * q + 0.3 * q**3)
        for q in qs
    ]


def test_grouped_step_matches_independent_ase_and_batches_eigh():
    qs, states = setup()
    optimizers = []
    for q in qs:
        atoms = Atoms("H" * len(q), positions=q.numpy())
        atoms.calc = Polynomial()
        optimizers.append(BFGS(atoms, logfile=None, alpha=53.0, maxstep=0.12))
    original = torch.linalg.eigh
    shapes = []

    def eigh(h, **kwargs):
        shapes.append(tuple(h.shape))
        return original(h, **kwargs)

    with patch("torch.linalg.eigh", side_effect=eigh):
        for _ in range(8):
            fs = forces(qs)
            saved = [f.clone() for f in fs]
            diagnostics = bfgs_ase_batch_step(qs, fs, states)
            for q, state, opt, result, f, old in zip(
                qs, states, optimizers, diagnostics, fs, saved
            ):
                opt.step()
                np.testing.assert_allclose(
                    q.numpy(), opt.atoms.positions, atol=3e-12, rtol=3e-12
                )
                np.testing.assert_allclose(
                    state.hessian.numpy(), opt.H, atol=3e-10, rtol=3e-11
                )
                assert torch.equal(f, old)
                assert result.eigenvalues.shape == (q.numel(),)
    assert shapes == [(2, 6, 6), (1, 9, 9)] * 8


def test_failure_in_later_group_does_not_commit_any_system():
    qs, states = setup()
    states[1].hessian.zero_()
    before = [q.clone() for q in qs]
    with pytest.raises(BFGSNumericalError, match="zero eigenvalue"):
        bfgs_ase_batch_step(qs, forces(qs), states)
    assert all(torch.equal(a, b) for a, b in zip(qs, before))
    assert all(s.step_count == 0 and not s.initialized for s in states)


def test_finite_incorrect_solver_output_is_rejected_atomically():
    qs, states = setup()
    before = [q.clone() for q in qs]
    original = torch.linalg.eigh

    def wrong(h, **kwargs):
        w, v = original(h, **kwargs)
        return w * 2, v

    with (
        patch("torch.linalg.eigh", side_effect=wrong),
        pytest.raises(BFGSNumericalError, match="invalid eigensystem"),
    ):
        bfgs_ase_batch_step(qs, forces(qs), states)
    assert all(torch.equal(a, b) for a, b in zip(qs, before))
    assert all(s.step_count == 0 for s in states)


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", "torch_reference"),
        pytest.param("cuda", "hip", marks=pytest.mark.gpu),
        pytest.param("cuda", "hip.bfgs-ase-jacobi-streams-v1", marks=pytest.mark.gpu),
    ],
)
def test_overflowed_step_length_matches_single_and_preserves_all_state(device, backend):
    """A finite direction whose norm overflows must fail before any group commits."""
    if device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        pytest.skip("HCU unavailable")
    qs = [torch.zeros(n, 3, dtype=torch.float64, device=device) for n in (1, 2)]
    fs = [torch.ones_like(qs[0]), torch.zeros_like(qs[1])]
    fs[1][0, 0] = 1e8
    states = [
        bfgs_ase_prepare_state(
            q.numel(), alpha=alpha, device=device, system_id=i, backend=backend
        )
        for i, (q, alpha) in enumerate(zip(qs, (70.0, 1e-300)))
    ]
    before = [state.state_dict() for state in states]
    coordinates, forces_before = [q.clone() for q in qs], [f.clone() for f in fs]
    single = ASEBFGSState.from_state_dict(before[1])
    with pytest.raises(BFGSNumericalError, match="non-finite step length"):
        bfgs_ase_step(qs[1].clone(), fs[1].clone(), single, backend=backend)
    with pytest.raises(BFGSNumericalError, match="non-finite step length"):
        bfgs_ase_batch_step(qs, fs, states, backend=backend)
    assert all(torch.equal(a, b) for a, b in zip(qs, coordinates))
    assert all(torch.equal(a, b) for a, b in zip(fs, forces_before))
    for state, saved in zip([*states, single], [*before, before[1]]):
        for key, value in state.state_dict().items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, saved[key])
            else:
                assert value == saved[key]


@pytest.mark.parametrize("bad", ["empty", "duplicate", "alias", "dtype", "length"])
def test_invalid_batch_contract(bad):
    qs, states = setup()
    fs = forces(qs)
    if bad == "empty":
        qs, fs, states = [], [], []
    if bad == "duplicate":
        states[2].system_id = 0
    if bad == "alias":
        fs[0] = qs[0]
    if bad == "dtype":
        qs[0] = qs[0].float()
    if bad == "length":
        fs.pop()
    with pytest.raises((ValueError, TypeError)):
        bfgs_ase_batch_step(qs, fs, states)
