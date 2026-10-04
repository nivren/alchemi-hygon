# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress generation, parameter gradients and backend wiring without dynamics."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest
import torch
from nvalchemiops.backend import (
    DEFAULT_IMPLEMENTATION_REGISTRY,
    BackendUnavailableError,
    Implementation,
    resolve_backend,
)
from nvalchemiops.executor import clear_entrypoint_cache

from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force
from nvalchemi.models._utils import apply_strain, autograd_stresses, prepare_strain
from nvalchemi.training.losses.terms import StressMSELoss


def harmonic_inputs(dtype=torch.float64):
    """A ragged two/three-bond Batch with nonorthogonal row lattices."""
    h = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.4, 2.5, 0.0], [-0.2, 0.3, 3.0]],
            [[2.2, 0.3, -0.2], [0.5, 2.7, 0.1], [0.3, -0.4, 3.1]],
        ],
        dtype=dtype,
    )
    q = torch.tensor(
        [
            [1.1, 0.2, -0.1],
            [0.2, 0.7, 0.4],
            [0.1, -0.3, 1.2],
            [0.8, 0.2, 0.1],
            [0.3, -0.6, 0.2],
        ],
        dtype=dtype,
    )
    idx = torch.tensor([0, 0, 1, 1, 1])
    return torch.einsum("ni,nij->nj", q, h[idx]), h, idx


def test_old_import_path_and_torch_ops_registration_without_warp():
    """Old import path remains valid and reference never imports Warp/barostats."""
    code = """
import sys
import torch
from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force, npt_cell_update
assert 'warp' not in sys.modules
assert 'nvalchemiops.dynamics.integrators' not in sys.modules
h = torch.eye(3, dtype=torch.float64)[None]
s = h.clone()
v = torch.ones(1, dtype=torch.float64)
assert torch.equal(stress_to_cell_force(s,h,v,backend='torch_reference'), -s)
assert 'warp' not in sys.modules
assert callable(npt_cell_update)
assert hasattr(torch.ops.nvalchemi, 'stress_to_cell_force')
"""
    result = subprocess.run(  # noqa: S603 - fixed import-only script
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_existing_strain_stress_and_cell_force_against_independent_energy(dtype):
    """Reused strain tools agree with harmonic virial and fixed-fraction cell gradient."""
    r, h, idx = harmonic_inputs(dtype)
    scaled_r, _, strain = prepare_strain(r, h, idx)
    energy = 0.5 * scaled_r.square().sum()
    stress = autograd_stresses(energy, strain, h, 2)
    virial = torch.zeros_like(h).index_add_(0, idx, r[:, :, None] * r[:, None, :])
    v = torch.linalg.det(h).abs()
    tol = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(stress, virial / v[:, None, None], rtol=tol, atol=tol)
    force = stress_to_cell_force(stress, h, v, False, backend="torch_reference")
    frac = torch.linalg.solve(h[idx].mT, r[:, :, None]).squeeze(-1)
    expected = -torch.zeros_like(h).index_add_(0, idx, frac[:, :, None] * r[:, None, :])
    torch.testing.assert_close(force, expected, rtol=tol, atol=tol)
    if dtype == torch.float64:
        fd = torch.zeros_like(h)
        for b in range(2):
            for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
                delta = torch.zeros_like(h)
                delta[b, i, j] = 1e-6
                if i != j:
                    delta[b, j, i] = 1e-6
                rp, _ = apply_strain(r, h, idx, delta)
                rm, _ = apply_strain(r, h, idx, -delta)
                derivative = (0.5 * rp.square().sum() - 0.5 * rm.square().sum()) / 2e-6
                fd[b, i, j] = derivative / v[b] / (2 if i != j else 1)
                fd[b, j, i] = fd[b, i, j]
        torch.testing.assert_close(stress, fd, rtol=1e-8, atol=1e-8)


def test_stress_loss_mixed_second_derivative_reaches_model_parameter():
    """Actual upstream stress loss must backpropagate through dE/dstrain to theta."""
    r, h, idx = harmonic_inputs()
    theta = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.float64))
    scaled_r, _, strain = prepare_strain(r, h, idx)
    energy = theta * 0.5 * scaled_r.square().sum()
    stress = autograd_stresses(energy, strain, h, 2, training=True)
    loss = StressMSELoss()(stress, torch.zeros_like(stress))
    loss.backward()
    v = torch.linalg.det(h).abs()
    base = (
        torch.zeros_like(h).index_add_(0, idx, r[:, :, None] * r[:, None, :])
        / v[:, None, None]
    )
    torch.testing.assert_close(theta.grad, 2 * theta.detach() * base.square().mean())
    assert theta.grad.abs() > 0


def test_pre_resolved_selection_is_forwarded_without_second_resolution(monkeypatch):
    """A frozen selection reaches the declared executor unchanged."""
    import nvalchemi._backend as backend_module

    r, h, _ = harmonic_inputs()
    s = h.clone()
    v = torch.linalg.det(h).abs()
    selected = resolve_backend(
        "torch_reference",
        operation="stress_to_cell_force",
        device="cpu",
        dtype=torch.float64,
        gradient_order=2,
        features={"row_cell"},
    )
    monkeypatch.setattr(
        backend_module, "resolve_backend", lambda *a, **k: pytest.fail("resolved twice")
    )
    result = stress_to_cell_force(s, h, v, False, selection=selected)
    torch.testing.assert_close(result, -v[:, None, None] * torch.linalg.solve(h.mT, s))


@pytest.mark.parametrize("backend", ["hip", "triton", "not-a-backend"])
def test_unsupported_backend_never_falls_back(backend):
    """No optimized stress conversion is registered in this slice."""
    _, h, _ = harmonic_inputs()
    with pytest.raises(BackendUnavailableError):
        stress_to_cell_force(h, h, torch.ones(2, dtype=h.dtype), backend=backend)


def test_legacy_default_selection_does_not_route_to_reference(monkeypatch):
    """No-Warp installations still honor None/warp rather than silently falling back."""
    import nvalchemi.dynamics._ops.stress as dispatch_module

    _, h, _ = harmonic_inputs()
    calls = []

    def legacy(*args):
        calls.append(args)
        return torch.full_like(args[0], 23)

    monkeypatch.setattr(dispatch_module, "_stress_to_cell_force_warp", legacy)
    for backend in (None, "warp"):
        result = stress_to_cell_force(
            h, h, torch.ones(2, dtype=h.dtype), backend=backend
        )
        assert torch.all(result == 23)
    assert len(calls) == 2


def test_wrong_operation_selection_is_rejected():
    """Reject a frozen selection belonging to another operation."""
    _, h, _ = harmonic_inputs()
    selection = resolve_backend(None, operation="fire2")
    with pytest.raises(ValueError, match="must target operation"):
        stress_to_cell_force(h, h, torch.ones(2, dtype=h.dtype), selection=selection)


def test_second_implementation_needs_only_registration_and_module(monkeypatch):
    """Exercise a different executor through the unchanged public dispatcher."""
    module = ModuleType("stress_contract_second_executor")
    calls = []

    def convert(stress, cell, volume, keep_aligned=True):
        calls.append(keep_aligned)
        return torch.full_like(stress, 37)

    module.stress_to_cell_force = convert
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import nvalchemiops.backend as backend_module
    from nvalchemiops.backend import ImplementationRegistry

    registry = ImplementationRegistry(DEFAULT_IMPLEMENTATION_REGISTRY.implementations())
    registry.register(
        Implementation(
            implementation_id="test.stress-second-v1",
            operation="stress_to_cell_force",
            family="test",
            executor=module.__name__,
            entrypoints=("stress_to_cell_force",),
            executor_owner="ops",
            features=frozenset({"row_cell"}),
            max_gradient_order=2,
            default_strategy=True,
        )
    )
    monkeypatch.setattr(backend_module, "DEFAULT_IMPLEMENTATION_REGISTRY", registry)
    clear_entrypoint_cache()
    try:
        _, h, _ = harmonic_inputs()
        result = stress_to_cell_force(
            h, h, torch.ones(2, dtype=h.dtype), False, backend="test"
        )
        assert torch.all(result == 37) and calls == [False]
    finally:
        clear_entrypoint_cache()
