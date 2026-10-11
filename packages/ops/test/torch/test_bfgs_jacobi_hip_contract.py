# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU/fake gates for explicit Jacobi dispatch and shared joint-solver atomicity."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nvalchemiops import _bfgs_jacobi_hip as solver
from nvalchemiops.backend import (
    DEFAULT_IMPLEMENTATION_REGISTRY,
    BackendUnavailableError,
    ImplementationRegistry,
    resolve_backend,
)
from nvalchemiops.bfgs_ase import BFGSNumericalError, bfgs_ase_prepare_state
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh

BACKEND = "hip.bfgs-ase-jacobi-streams-v1"


def test_exact_selection_keeps_existing_defaults_and_rejects_capability():
    """Only exact opt-in selects Jacobi workers; hip/default/auto stay unchanged."""
    kw = dict(
        operation="bfgs_ase_compat",
        device="cuda",
        dtype=torch.float64,
        features={"batch", "heterogeneous", "variable_cell", "ase_unitcell"},
    )
    chosen = resolve_backend(BACKEND, **kw)
    assert chosen.implementation_id == BACKEND
    assert chosen.strategy == "jacobi_streams"
    # Explicit-only admission must not rely on the catalog's registration order.
    new = DEFAULT_IMPLEMENTATION_REGISTRY.get(BACKEND)
    old = DEFAULT_IMPLEMENTATION_REGISTRY.get("hip.bfgs-ase-rocsolver-v1")
    reordered = ImplementationRegistry((new, old))
    assert reordered.resolve("hip", **kw).implementation_id == old.implementation_id
    assert resolve_backend("hip", **kw).implementation_id == "hip.bfgs-ase-rocsolver-v1"
    assert (
        resolve_backend("torch_reference", **kw).implementation_id
        == "torch_reference.bfgs-ase-v1"
    )
    assert resolve_backend(None, **kw).family == "warp"
    with pytest.raises(BackendUnavailableError):
        resolve_backend("auto", **kw)
    for change in (
        dict(device="cpu"),
        dict(dtype=torch.float32),
        dict(gradient_order=1),
        dict(features={"six_cell_dofs"}),
    ):
        with pytest.raises(BackendUnavailableError):
            resolve_backend(BACKEND, **(kw | change))


@pytest.mark.parametrize("shape", [(1, 1), (513, 513), (32, 189, 189), (2**31, 1, 1)])
def test_fake_dynamic_outputs_do_not_load_runtime(shape, monkeypatch):
    """Fake output metadata covers previously unsupported widths/batches."""
    monkeypatch.setattr(
        solver, "_load_extension", lambda *a: pytest.fail("native loaded")
    )
    with FakeTensorMode():
        a = torch.empty(shape, device="cuda", dtype=torch.float64)
        w, v, info, residual, sweeps = solver._jacobi_eigh(a, 100)
        assert w.shape == a.shape[:-1] and v.shape == a.shape
        assert (
            info.shape
            == sweeps.shape
            == residual.shape
            == (shape[0] if len(shape) == 3 else 1,)
        )
        assert w.dtype == v.dtype == residual.dtype == torch.float64
        assert info.dtype == sweeps.dtype == torch.int32
        assert all(
            t.device == a.device and t.is_contiguous()
            for t in (w, v, info, residual, sweeps)
        )


@pytest.mark.parametrize(
    "case", ["rank", "square", "empty", "batch_empty", "dtype", "layout", "sweeps"]
)
def test_fake_invalid_metadata_rejected(case):
    """Unsupported shapes/dtypes/layouts are never converted implicitly."""
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
        if case == "layout":
            a = a.mT
        with pytest.raises(ValueError):
            solver._jacobi_eigh(a, 101 if case == "sweeps" else 100)


def test_cpu_gradient_upper_empty_rejected_before_native(monkeypatch):
    """No CPU fallback, autograd, upper triangle or empty private group operation."""
    monkeypatch.setattr(
        solver, "_load_extension", lambda *a: pytest.fail("native loaded")
    )
    with pytest.raises(BackendUnavailableError, match="HIP device"):
        solver._eigh(torch.eye(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="forward-only"):
        solver._validate(
            torch.ones(3, 3, dtype=torch.float64, requires_grad=True), "L", 100
        )
    with pytest.raises(ValueError, match="nonempty"):
        solver._grouped_eigh([])
    with FakeTensorMode():
        a = torch.empty(3, 3, device="cuda", dtype=torch.float64)
        with pytest.raises(ValueError, match="LOWER"):
            solver._eigh(a, "U")


@pytest.mark.parametrize("failure", ["late_group", "incomplete", "invalid_vector"])
def test_joint_failure_does_not_commit_any_group(failure):
    """Candidate Hessians may change; all persistent histories remain atomic."""
    qs = [torch.full((n, 3), 0.3, dtype=torch.float64) for n in (1, 2)]
    states = [bfgs_ase_prepare_state(q.numel(), system_id=i) for i, q in enumerate(qs)]
    _step_batch_with_eigh(qs, [-q for q in qs], states, torch.linalg.eigh)
    before_q = [q.clone() for q in qs]
    before = [s.state_dict() for s in states]

    def joint(hs):
        assert all(not torch.equal(h, s.hessian) for h, s in zip(hs, states))
        out = [torch.linalg.eigh(h) for h in hs]
        if failure == "late_group":
            raise BFGSNumericalError("last group failed")
        if failure == "incomplete":
            return out[:1]
        out[-1] = (out[-1][0], torch.zeros_like(out[-1][1]))
        return out

    with pytest.raises(BFGSNumericalError):
        _step_batch_with_eigh(
            qs, [-q for q in qs], states, torch.linalg.eigh, grouped_eigh=joint
        )
    assert all(torch.equal(q, old) for q, old in zip(qs, before_q))
    for state, expected in zip(states, before):
        for key, value in expected.items():
            actual = state.state_dict()[key]
            assert (
                torch.equal(actual, value)
                if isinstance(value, torch.Tensor)
                else actual == value
            )


def test_single_width_avoids_worker_creation(monkeypatch):
    """A direct batched solve does not allocate a pool or split its batch."""
    sentinel = object()
    monkeypatch.setattr(solver, "_validate", lambda *a: None)
    monkeypatch.setattr(solver, "_eigh", lambda a: sentinel)
    monkeypatch.setattr(
        solver, "_GroupScheduler", lambda *a: pytest.fail("worker created")
    )
    assert solver._grouped_eigh([torch.eye(3)]) == [sentinel]


def test_unqualified_architecture_and_failed_build_do_not_cache_or_fallback(
    monkeypatch,
):
    """Wrong architecture and build exceptions fail before any alternative solver."""
    from nvalchemiops import _hip_extension_loader as loader

    monkeypatch.setattr(solver, "_EXTENSION", None)
    monkeypatch.setattr(solver, "_check_runtime", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args: SimpleNamespace(gcnArchName="gfx999"),
    )
    monkeypatch.setattr(
        loader,
        "load_hip_extension",
        lambda **kwargs: pytest.fail("unqualified build attempted"),
    )
    with pytest.raises(BackendUnavailableError, match="gfx936"):
        solver._load_extension(torch.device("cuda", 0))
    assert solver._EXTENSION is None
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args: SimpleNamespace(gcnArchName="gfx936"),
    )

    def fail(**kwargs):
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(loader, "load_hip_extension", fail)
    with pytest.raises(RuntimeError, match="build failure"):
        solver._load_extension(torch.device("cuda", 0))
    assert solver._EXTENSION is None
