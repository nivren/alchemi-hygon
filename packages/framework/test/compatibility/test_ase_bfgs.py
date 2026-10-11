# SPDX-License-Identifier: Apache-2.0
"""ASE oracle and lifecycle gates for the local dense BFGS optimizer."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from nvalchemiops.backend import BackendUnavailableError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGS
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.models.base import BaseModelMixin, ModelConfig


class PolynomialModel(torch.nn.Module, BaseModelMixin):
    """Independent diagonal polynomial with explicit forces and call counts."""

    def __init__(self, quartic=0.0):
        super().__init__()
        self.quartic = quartic
        self.calls = 0
        self.model_config = ModelConfig(outputs={"energy", "forces"})

    def forward(self, batch):
        self.calls += 1
        q = batch.positions
        k = torch.arange(1, q.numel() + 1, dtype=q.dtype, device=q.device).view_as(q)
        return OrderedDict(
            energy=(0.5 * k * q**2 + self.quartic * q**4 / 4).sum().reshape(1, 1),
            forces=-(k * q + self.quartic * q**3),
        )

    def adapt_input(self, data, **kwargs):
        return data

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        return data

    def adapt_output(self, output, **kwargs):
        return output


class PolynomialCalculator(Calculator):
    """NumPy oracle independent of the Torch framework model."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, quartic=0.0):
        super().__init__()
        self.quartic = quartic

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        q = self.atoms.positions
        k = np.arange(1, q.size + 1).reshape(q.shape)
        self.results = dict(
            energy=(0.5 * k * q**2 + self.quartic * q**4 / 4).sum(),
            forces=-(k * q + self.quartic * q**3),
        )


def make_batch(n=3):
    """Construct one system with deliberately stale initial outputs."""
    q = torch.linspace(-0.4, 0.6, 3 * n, dtype=torch.float64).reshape(n, 3)
    b = Batch.from_data_list(
        [
            AtomicData(
                positions=q,
                atomic_numbers=torch.ones(n, dtype=torch.long),
                forces=torch.zeros_like(q),
                energy=torch.zeros(1, 1, dtype=q.dtype),
                cell=torch.eye(3, dtype=q.dtype)[None] * 10,
                pbc=torch.zeros(1, 3, dtype=torch.bool),
            )
        ]
    )
    b.system_id = torch.tensor([[17]])
    return b


def test_public_lazy_exports_do_not_import_warp_or_eager_executor():
    code = """
import sys
from nvalchemi.dynamics import ASECompatibleBFGS
from nvalchemi.dynamics.optimizers import ASECompatibleBFGS as second
assert second is ASECompatibleBFGS
assert 'warp' not in sys.modules
assert 'nvalchemiops.bfgs_ase_reference' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)  # noqa: S603 - fixed import-only code


@pytest.mark.parametrize("quartic", [0.0, 0.3])
@pytest.mark.parametrize("n", [1, 3, 12])
def test_public_step_matches_ase_hessian_direction_and_positions(quartic, n):
    b = make_batch(n)
    atoms = Atoms("H" * n, positions=b.positions.numpy().copy())
    atoms.calc = PolynomialCalculator(quartic)
    ase = BFGS(atoms, logfile=None, maxstep=0.12, alpha=53.0)
    model = PolynomialModel(quartic)
    d = ASECompatibleBFGS(model, maxstep=0.12, alpha=53.0, fmax=1e-10)
    for i in range(8):
        before = b.positions.clone()
        ase.step()
        d.step(b)
        np.testing.assert_allclose(
            b.positions.numpy(), atoms.positions, atol=2e-12, rtol=2e-12
        )
        np.testing.assert_allclose(
            d.bfgs_state.hessian.numpy(), ase.H, atol=2e-10, rtol=2e-11
        )
        torch.testing.assert_close(
            d.last_step.applied_step, b.positions - before, atol=1e-16, rtol=1e-14
        )
        assert d.bfgs_state.step_count == i + 1
        assert (
            model.calls == i + 2
        )  # Initial force plus one new-position evaluation per update.


@pytest.mark.parametrize(
    "initial_force,updates", [(0.0, 0), (0.049, 0), (0.05, 1), (0.051, 1)]
)
def test_initial_convergence_strict_boundary(initial_force, updates):
    b = make_batch(1)
    b.positions.zero_()
    b.positions[0, 0] = initial_force
    model = PolynomialModel()
    d = ASECompatibleBFGS(model)
    d.run(b, n_steps=1)
    assert d.bfgs_state.step_count == updates
    assert d.step_count == updates
    assert model.calls == updates + 1


def test_stop_counts_and_hook_order_match_ase():
    b = make_batch(3)
    atoms = Atoms("H3", positions=b.positions.numpy().copy())
    atoms.calc = PolynomialCalculator(0.3)
    ase = BFGS(atoms, logfile=None)
    ase.run(fmax=0.05, steps=100)
    calls = []

    class RecordingHook:
        frequency = 1

        def __init__(self, stage):
            self.stage = stage
            self.open, self.close = Mock(), Mock()

        def __call__(self, ctx, stage):
            calls.append(stage)

        def __enter__(self):
            self.open()
            return self

        def __exit__(self, *args):
            self.close()

    hooks = [RecordingHook(stage) for stage in DynamicsStage]
    model = PolynomialModel(0.3)
    d = ASECompatibleBFGS(model, hooks=hooks)
    d.run(b, n_steps=100)
    assert d.bfgs_state.step_count == ase.nsteps == d.step_count
    assert model.calls == ase.nsteps + 1
    np.testing.assert_allclose(b.positions.numpy(), atoms.positions, atol=2e-11)
    sequence = [
        DynamicsStage.BEFORE_STEP,
        DynamicsStage.BEFORE_PRE_UPDATE,
        DynamicsStage.AFTER_PRE_UPDATE,
        DynamicsStage.BEFORE_COMPUTE,
        DynamicsStage.AFTER_COMPUTE,
        DynamicsStage.BEFORE_POST_UPDATE,
        DynamicsStage.AFTER_POST_UPDATE,
        DynamicsStage.AFTER_STEP,
    ]
    assert calls[:2] == [DynamicsStage.BEFORE_COMPUTE, DynamicsStage.AFTER_COMPUTE]
    assert calls[2:10] == sequence
    assert calls[-1] == DynamicsStage.ON_CONVERGE
    assert all(hook.open.call_count == hook.close.call_count == 1 for hook in hooks)


@pytest.mark.parametrize(
    "method", ["step", "masked_update", "_masked_pre_update", "_masked_post_update"]
)
def test_all_inactive_entrances_freeze_full_history(method):
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel(), fmax=1e-10)
    d.run(b, n_steps=2)
    saved = d.bfgs_state.state_dict()
    old = b.clone()
    if method == "step":
        b.status = torch.tensor([[d.exit_status]])
        d.step(b)
    else:
        getattr(d, method)(b, torch.tensor([False]))
    for key, value in saved.items():
        actual = d.bfgs_state.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual, value), key
        else:
            assert actual == value, key
    for key in ("positions", "forces", "energy", "cell"):
        assert torch.equal(getattr(b, key), getattr(old, key)), key


def test_restart_with_fresh_force_recomputation_matches_uninterrupted():
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel(0.3), alpha=53, maxstep=0.12, fmax=1e-10)
    d.run(b, n_steps=4)
    geometry = b.clone()
    saved = d.state_dict()
    d.run(b, n_steps=4)
    resumed = ASECompatibleBFGS(PolynomialModel(0.3))
    resumed.load_state_dict(saved)
    geometry.forces.fill_(
        123
    )  # Restart must recompute, not consume stale caller outputs.
    resumed.run(geometry, n_steps=4)
    torch.testing.assert_close(geometry.positions, b.positions, atol=0, rtol=0)
    torch.testing.assert_close(
        resumed.bfgs_state.hessian, d.bfgs_state.hessian, atol=0, rtol=0
    )
    assert resumed.step_count == d.step_count == 8
    assert resumed.model.calls == 5
    saved["bfgs_state"]["hessian"].zero_()
    assert resumed.bfgs_state.hessian.abs().sum() > 0


@pytest.mark.parametrize("change", ["cell", "pbc", "atomic_numbers", "system_id"])
def test_identity_or_fixed_cell_changes_rejected(change):
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel())
    d.step(b)
    value = getattr(b, change)
    if value.dtype == torch.bool:
        value.logical_not_()
    else:
        value.add_(1)
    with pytest.raises(ValueError, match="identity|fixed|system_id"):
        d.step(b)


@pytest.mark.parametrize("backend", [None, "warp", "auto", "hip", "triton", "unknown"])
def test_unsupported_backend_fails_explicitly(backend):
    with pytest.raises(BackendUnavailableError):
        ASECompatibleBFGS(PolynomialModel(), backend=backend).step(make_batch())


def test_framework_resolves_once_and_passes_cached_selection(monkeypatch):
    import nvalchemiops.bfgs_ase as ops

    import nvalchemi._backend as backend

    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel())
    d.step(b)
    monkeypatch.setattr(
        backend, "resolve_backend", lambda *a, **k: pytest.fail("resolved twice")
    )
    monkeypatch.setattr(
        ops, "resolve_backend", lambda *a, **k: pytest.fail("Ops resolved again")
    )
    d.step(b)


@pytest.mark.parametrize("replacement", [False, True])
def test_external_coordinate_mutation_or_replacement_recomputes_forces(replacement):
    b = make_batch()
    model = PolynomialModel()
    d = ASECompatibleBFGS(model)
    d.step(b)
    if replacement:
        b.positions = b.positions.clone() + 0.1
    else:
        b.positions.add_(0.1)
    d.step(b)
    assert (
        model.calls == 4
    )  # Second call primes changed coordinates, then computes new forces.


def test_masks_validate_before_update_and_active_mask_uses_ase_step():
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel())
    with pytest.raises(ValueError, match="mask"):
        d._masked_pre_update(b, torch.tensor([1]))
    old = b.positions.clone()
    d._masked_pre_update(b, torch.tensor([True]))
    k = torch.arange(1, b.positions.numel() + 1).reshape_as(old)
    torch.testing.assert_close(
        d.last_step.raw_step, -k * old / 70, atol=1e-15, rtol=1e-14
    )
    assert d.bfgs_state.step_count == 1


def test_restart_rejects_wrong_paired_geometry():
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel())
    d.step(b)
    resumed = ASECompatibleBFGS(PolynomialModel())
    resumed.load_state_dict(d.state_dict())
    b.positions.add_(0.001)
    with pytest.raises(ValueError, match="paired Batch"):
        resumed.step(b)


def test_direct_masked_update_snapshot_preserves_independent_counts():
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel())
    d._masked_pre_update(b, torch.tensor([True]))
    assert d.step_count == 0 and d.bfgs_state.step_count == 1
    resumed = ASECompatibleBFGS(PolynomialModel())
    resumed.load_state_dict(d.state_dict())
    resumed.step(b)
    assert resumed.step_count == 1 and resumed.bfgs_state.step_count == 2


def test_restart_does_not_silently_replace_custom_stopping():
    hook = ConvergenceHook(
        criteria={"key": "forces", "threshold": 0.001, "reduce_op": "norm"}
    )
    d = ASECompatibleBFGS(PolynomialModel(), convergence_hook=hook)
    d.step(make_batch())
    with pytest.raises(ValueError, match="convergence mode"):
        ASECompatibleBFGS(PolynomialModel()).load_state_dict(d.state_dict())


@pytest.mark.parametrize("field", ["forces", "energy"])
def test_output_alias_is_rejected_before_initial_model_mutation(field):
    b = make_batch()
    before = b.positions.clone()
    setattr(
        b,
        field,
        b.positions if field == "forces" else b.positions.flatten()[:1].reshape(1, 1),
    )
    model = PolynomialModel()
    with pytest.raises(ValueError, match="share storage"):
        ASECompatibleBFGS(model).step(b)
    assert model.calls == 0 and torch.equal(b.positions, before)


def test_custom_convergence_and_invalid_modes():
    hook = ConvergenceHook(
        criteria={"key": "forces", "threshold": 0.001, "reduce_op": "norm"}
    )
    d = ASECompatibleBFGS(PolynomialModel(), convergence_hook=hook)
    assert d.convergence_hook is hook
    b = make_batch()
    b = Batch.from_data_list(b.to_data_list() * 2)
    with pytest.raises(ValueError, match="unique"):
        d.step(b)
    # M4 retains native sampler configuration; duplicated IDs fail admission.
    sampler = object()
    assert ASECompatibleBFGS(PolynomialModel(), sampler=sampler).sampler is sampler
    with pytest.raises(ValueError, match="float64"):
        bad = make_batch()
        bad.positions = bad.positions.float()
        ASECompatibleBFGS(PolynomialModel()).step(bad)


@pytest.mark.parametrize(
    "key,bad", [("version", 2), ("step_count", -1), ("step_count", True), ("fmax", 0)]
)
def test_restart_validation_is_atomic(key, bad):
    b = make_batch()
    d = ASECompatibleBFGS(PolynomialModel())
    d.step(b)
    before = d.state_dict()
    invalid = copy.deepcopy(before)
    invalid[key] = bad
    with pytest.raises((ValueError, TypeError)):
        d.load_state_dict(invalid)
    assert torch.equal(before["bfgs_state"]["hessian"], d.bfgs_state.hessian)
