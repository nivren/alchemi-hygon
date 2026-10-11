# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded single-HCU numerical, failure, stream and custom-op acceptance."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops import _bfgs_eigh_hip as solver
from nvalchemiops.bfgs_ase import (
    BFGSNumericalError,
    bfgs_ase_prepare_state,
    bfgs_ase_step,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.version.hip or not torch.cuda.is_available(),
        reason="requires visible HIP device",
    ),
]


def symmetric(n, count=1):
    """Independent CPU random input; setup copies are outside the solver."""
    generator = torch.Generator().manual_seed(24 + n + count)
    cpu = torch.randn(count, n, n, dtype=torch.float64, generator=generator)
    return (cpu + cpu.mT) * 0.5


def check_eigensystem(a, w, v, cpu, *, normalized_orthogonality=False):
    """Check spectral values and invariants independently of vector signs."""
    oracle, _ = torch.linalg.eigh(cpu)
    torch.testing.assert_close(w.cpu(), oracle, atol=2e-10, rtol=2e-11)
    residual = ((v * w.unsqueeze(-2)) @ v.mT - a).norm(dim=(-1, -2)) / a.norm(
        dim=(-1, -2)
    ).clamp_min(1)
    orthogonal = (
        v.mT @ v - torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    ).norm(dim=(-1, -2))
    assert float(residual.max()) < 1e-11
    if normalized_orthogonality:
        # Large repeated eigenspaces: Frobenius error normalized by ||I||_F.
        # Retain the existing absolute bound for all original random tests.
        assert float(orthogonal.max()) / a.shape[-1] ** 0.5 < 1e-12
    else:
        assert float(orthogonal.max()) < 1e-11
    assert w.device == v.device == a.device and w.dtype == v.dtype == torch.float64
    assert w.is_contiguous() and v.is_contiguous()
    assert (
        a.data_ptr() not in (w.data_ptr(), v.data_ptr())
        and w.data_ptr() != v.data_ptr()
    )


@pytest.mark.parametrize("n", [1, 3, 9, 36, 138, 199, 200, 201, 552])
@pytest.mark.parametrize("batched", [False, True])
def test_symmetric_eigensystem_and_input_preservation(n, batched):
    """Small/boundary/large and batch layouts preserve FP64 input."""
    cpu = symmetric(n, 2 if batched else 1)
    if not batched:
        cpu = cpu[0]
    a = cpu.cuda()
    before = a.clone()
    w, v, info, residual, sweeps = solver._rocsolver_eigh(a, "L", 100)
    check_eigensystem(a, w, v, cpu)
    assert torch.equal(a, before) and not bool(info.any())
    if n < 200:
        assert bool((sweeps <= 100).all()) and bool(torch.isfinite(residual).all())


@pytest.mark.parametrize("n", [9, 200])
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_triangle_selection_ignores_other_triangle(n, uplo):
    """UPLO flips correctly for row storage, including an unused NaN triangle."""
    cpu = symmetric(n, 2)
    a = cpu.cuda()
    indices = (
        torch.triu_indices(n, n, 1, device=a.device)
        if uplo == "L"
        else torch.tril_indices(n, n, -1, device=a.device)
    )
    a[:, indices[0], indices[1]] = float("nan")
    before = a.clone()
    w, v = solver._eigh(a, UPLO=uplo)
    check_eigensystem(cpu.cuda(), w, v, cpu)
    torch.testing.assert_close(a, before, equal_nan=True, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["zero", "negative", "repeated", "nearzero"])
def test_special_spectra(kind):
    """No spectral damping/clamp and no assumptions of positive definiteness."""
    diagonals = {
        "zero": [0.0, 0.0, 0.0],
        "negative": [-3.0, -2.0, -1.0],
        "repeated": [-1.0, -1.0, 2.0],
        "nearzero": [1e-18, -1.0, 2.0],
    }
    cpu = torch.diag(torch.tensor(diagonals[kind], dtype=torch.float64))
    a = cpu.cuda()
    w, v = solver._eigh(a)
    check_eigensystem(a, w, v, cpu)
    torch.testing.assert_close(
        w.cpu(), torch.sort(cpu.diagonal()).values, atol=0, rtol=0
    )


@pytest.mark.parametrize("n", [200, 276, 285, 552])
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_large_repeated_alpha_low_rank_hessian(n, uplo):
    """Early BFGS retains a large repeated-alpha eigenspace, unlike random SPD."""
    generator = torch.Generator().manual_seed(24 + n)
    u = torch.randn(2, n, 18, dtype=torch.float64, generator=generator)
    cpu = 70 * torch.eye(n, dtype=torch.float64)[None] - 0.03 * (u @ u.mT)
    a = cpu.cuda()
    for _ in range(3):
        w, v = solver._eigh(a, UPLO=uplo)
        check_eigensystem(a, w, v, cpu, normalized_orthogonality=True)


def test_nondefault_stream_and_device_restoration():
    """Bind to the caller stream and consume results through an explicit wait."""
    original_device = torch.cuda.current_device()
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    cpu = symmetric(36, 4)
    with torch.cuda.stream(producer):
        a = cpu.cuda()
        a.mul_(2)
        w, v = solver._eigh(a)
    torch.cuda.current_stream().wait_stream(producer)
    assert torch.cuda.current_device() == original_device
    check_eigensystem(a, w, v, cpu * 2)


def test_schema_fake_opcheck_and_forward_only():
    """Schema/fake correctness is separately checked, without claiming compile."""
    a = symmetric(9)[0].cuda()
    result = torch.library.opcheck(
        solver._rocsolver_eigh,
        (a, "L", 100),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(value == "SUCCESS" for value in result.values())
    with pytest.raises(ValueError, match="forward-only"):
        solver._rocsolver_eigh(a.requires_grad_(), "L", 100)


def test_nonconvergence_info_and_atomic_bfgs_failure(monkeypatch):
    """A real one-sweep failure raises and cannot commit BFGS state."""
    a = symmetric(138)[0].cuda()
    with pytest.raises(BFGSNumericalError, match="solver info"):
        solver._rocsolver_eigh(a, "L", 1)
    state = bfgs_ase_prepare_state(138, device="cuda", backend="hip")
    state.hessian.copy_(a)
    q = torch.ones(46, 3, device="cuda", dtype=torch.float64)
    before_q, before = q.clone(), state.state_dict()
    extension = solver._load_extension()
    original = extension.eigh
    monkeypatch.setattr(extension, "eigh", lambda x, uplo, sweeps: original(x, uplo, 1))
    with pytest.raises(BFGSNumericalError, match="solver info"):
        bfgs_ase_step(q, -q, state, backend="hip")
    assert torch.equal(q, before_q)
    for key, value in before.items():
        current = state.state_dict()[key]
        assert (
            torch.equal(value, current)
            if isinstance(value, torch.Tensor)
            else value == current
        )


def test_bad_selected_triangle_and_exact_zero_bfgs_rejected():
    """Nonfinite inputs and ASE exact singular directions fail explicitly."""
    a = torch.eye(3, device="cuda", dtype=torch.float64)
    a[1, 0] = float("nan")
    with pytest.raises(BFGSNumericalError, match="selected triangle"):
        solver._eigh(a)
    state = bfgs_ase_prepare_state(3, device="cuda", backend="hip")
    state.hessian.zero_()
    q = torch.ones(1, 3, device="cuda", dtype=torch.float64)
    with pytest.raises(BFGSNumericalError, match="exact zero eigenvalue"):
        bfgs_ase_step(q, -q, state, backend="hip")
    assert not state.initialized and state.step_count == 0


def test_native_api_error_status_is_propagated_before_commit(monkeypatch):
    """A library API failure cannot trigger a reference fallback."""
    state = bfgs_ase_prepare_state(3, device="cuda", backend="hip")
    q = torch.ones(1, 3, device="cuda", dtype=torch.float64)
    extension = solver._load_extension()

    def fail(*args):
        raise RuntimeError(
            "rocsolver_dsyevj_strided_batched failed with rocBLAS/rocSOLVER API status 5"
        )

    monkeypatch.setattr(extension, "eigh", fail)
    with pytest.raises(RuntimeError, match="API status 5"):
        bfgs_ase_step(q, -q, state, backend="hip")
    assert torch.equal(q, torch.ones_like(q)) and state.step_count == 0
