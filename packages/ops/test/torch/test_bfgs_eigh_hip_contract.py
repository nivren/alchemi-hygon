# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU/fake tests for the explicit native BFGS boundary and atomic failure."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nvalchemiops import _bfgs_eigh_hip as solver
from nvalchemiops.backend import BackendUnavailableError, resolve_backend
from nvalchemiops.bfgs_ase import BFGSNumericalError, bfgs_ase_prepare_state
from nvalchemiops.bfgs_ase_reference import _step_with_eigh


def test_native_registration_has_narrow_explicit_capability():
    """Select the native executor without loading or broadening capabilities."""
    chosen = resolve_backend(
        "hip",
        operation="bfgs_ase_compat",
        device="cuda",
        dtype=torch.float64,
        features={"single_system", "fixed_cell"},
    )
    assert chosen.implementation_id == "hip.bfgs-ase-rocsolver-v1"
    for kwargs in (
        {"device": "cpu"},
        {"dtype": torch.float32},
        {"gradient_order": 1},
        {"features": {"variable_cell", "six_cell_dofs"}},
        {"features": {"batched"}},
    ):
        options = dict(
            device="cuda", dtype=torch.float64, features={"single_system", "fixed_cell"}
        )
        options.update(kwargs)
        with pytest.raises(BackendUnavailableError):
            resolve_backend("hip", operation="bfgs_ase_compat", **options)


@pytest.mark.parametrize("shape", [(3, 3), (4, 9, 9)])
def test_fake_shapes_do_not_load_native_or_assume_runtime(shape, monkeypatch):
    """Fake outputs have independent contiguous storage and correct metadata."""
    monkeypatch.setattr(solver, "_load_extension", lambda: pytest.fail("native loaded"))
    with FakeTensorMode():
        a = torch.empty(shape, dtype=torch.float64, device="cuda")
        w, v, info, residual, sweeps = solver._rocsolver_eigh(a, "L", 100)
        assert w.shape == shape[:-1] and v.shape == shape
        assert w.dtype == v.dtype == residual.dtype == torch.float64
        assert info.dtype == sweeps.dtype == torch.int32
        assert info.shape == (1 if len(shape) == 2 else shape[0],)
        for result in (w, v, info, residual, sweeps):
            assert result.device == a.device and result.is_contiguous()


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "square",
        "empty",
        "batch_empty",
        "dtype",
        "layout",
        "grad",
        "uplo",
        "sweeps",
    ],
)
def test_fake_invalid_metadata_fails_without_native(case):
    """Unsupported input contracts never become a hidden conversion."""
    if case == "grad":
        # HIP Autograd device initialization is unnecessary for metadata checks.
        a = torch.empty((3, 3), dtype=torch.float64, requires_grad=True)
        with pytest.raises(ValueError, match="forward-only"):
            solver._validate(a, "L", 100)
        return
    with FakeTensorMode():
        shape = {
            "rank": (1, 1, 3, 3),
            "square": (3, 2),
            "empty": (0, 0),
            "batch_empty": (0, 3, 3),
        }.get(case, (3, 3))
        a = torch.empty(
            shape,
            device="cuda",
            dtype=torch.float32 if case == "dtype" else torch.float64,
        )
        uplo, sweeps = "L", 100
        if case == "layout":
            a = a.mT
        if case == "uplo":
            uplo = "lower"
        if case == "sweeps":
            sweeps = 0
        with pytest.raises(ValueError):
            solver._validate(a, uplo, sweeps)


def test_cpu_rejected_before_extension(monkeypatch):
    """An explicit HIP request cannot move a matrix to another device."""
    monkeypatch.setattr(solver, "_load_extension", lambda: pytest.fail("native loaded"))
    with pytest.raises(BackendUnavailableError, match="HIP device"):
        solver._eigh(torch.eye(3, dtype=torch.float64))


def test_unverified_build_rejected(monkeypatch):
    """A cuda label is insufficient to admit another solver library build."""
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(BackendUnavailableError, match="requalification"):
        solver._check_runtime()


def test_native_library_version_mismatch_and_build_error(monkeypatch):
    """Fail explicit loading rather than caching another implementation."""
    from types import SimpleNamespace

    from nvalchemiops import _hip_extension_loader as loader

    monkeypatch.setattr(solver, "_EXTENSION", None)
    monkeypatch.setattr(solver, "_check_runtime", lambda: None)
    monkeypatch.setattr(
        loader,
        "load_hip_extension",
        lambda **kw: SimpleNamespace(rocsolver_version=lambda: "9.0.0"),
    )
    with pytest.raises(BackendUnavailableError, match="got 9.0.0"):
        solver._load_extension()
    assert solver._EXTENSION is None

    def broken(**kwargs):
        raise RuntimeError("native build failed")

    monkeypatch.setattr(loader, "load_hip_extension", broken)
    with pytest.raises(RuntimeError, match="native build failed"):
        solver._load_extension()
    assert solver._EXTENSION is None


def test_eigh_failure_is_atomic_even_after_candidate_hessian_update():
    """Solver errors cannot commit candidate coordinates or updated history."""
    q = torch.tensor([[0.2, 0.3, -0.4]], dtype=torch.float64)
    state = bfgs_ase_prepare_state(3)
    _step_with_eigh(q, -q, state, torch.linalg.eigh)
    before_q, before = q.clone(), state.state_dict()

    def fail(a, UPLO):
        assert not torch.equal(a, state.hessian)
        raise BFGSNumericalError("non-converged")

    with pytest.raises(BFGSNumericalError, match="non-converged"):
        _step_with_eigh(q, -q, state, fail)
    assert torch.equal(q, before_q)
    for key, value in before.items():
        current = state.state_dict()[key]
        assert (
            torch.equal(value, current)
            if isinstance(value, torch.Tensor)
            else value == current
        )
