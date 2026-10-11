# SPDX-License-Identifier: Apache-2.0
"""Allocated-card HCU batching, stream and lifecycle acceptance."""

from unittest.mock import patch

import pytest
import torch

from nvalchemi.dynamics import ASECompatibleBFGS, ASECompatibleBFGSVariableCell
from test.compatibility.test_ase_bfgs_batch import (
    BatchCell,
    BatchPolynomial,
    check_single_restart_promotion,
    mixed,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not getattr(torch.version, "__hcu_version__", None),
    reason="requires an allocated HCU on the admitted Torch build",
)


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("backend", ["torch_reference", "hip"])
def test_explicit_restore_grouped_solver_stream_and_freeze(variable, backend):
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    model_cls = BatchCell if variable else BatchPolynomial
    cpu_batch = mixed(variable)
    cpu = cls(model_cls(), fmax=1e-10)
    cpu.run(cpu_batch, n_steps=2)
    gpu_batch = cpu_batch.clone().to("cuda")
    gpu = cls(model_cls(), backend=backend)
    gpu.load_state_dict(cpu.state_dict(), device="cuda")
    gpu._ensure_state_initialized(gpu_batch)
    shapes = []
    if backend == "hip":
        from nvalchemiops.bfgs_ase_hip import _eigh as original

        target = "nvalchemiops.bfgs_ase_hip._eigh"
    else:
        original, target = torch.linalg.eigh, "torch.linalg.eigh"

    def eigh(h, **kwargs):
        assert h.device == gpu_batch.positions.device and h.dtype == torch.float64
        shapes.append(tuple(h.shape))
        return original(h, **kwargs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), patch(target, side_effect=eigh):
        for _ in range(3):
            gpu.step(gpu_batch)
        snapshot = gpu.state_dict()
        paired = gpu_batch.clone()
        resumed = cls(model_cls(), backend=backend)
        resumed.load_state_dict(snapshot)
        gpu.step(gpu_batch)
        resumed.step(paired)
        done = torch.cuda.Event()
        done.record()
    torch.cuda.current_stream().wait_event(done)
    for _ in range(4):
        cpu.step(cpu_batch)
    torch.testing.assert_close(
        gpu_batch.positions.cpu(), cpu_batch.positions, atol=5e-12, rtol=5e-12
    )
    if variable:
        torch.testing.assert_close(
            gpu_batch.cell.cpu(), cpu_batch.cell, atol=5e-12, rtol=5e-12
        )
    torch.testing.assert_close(gpu_batch.positions, paired.positions, atol=0, rtol=0)
    width = 15 if variable else 6
    assert shapes == [(2, width, width), (1, width + 3, width + 3)] * 5
    old = gpu.bfgs_states[12].state_dict()
    saved = gpu_batch.clone()
    gpu._masked_pre_update(gpu_batch, torch.tensor([True, False, True], device="cuda"))
    assert torch.equal(gpu.bfgs_states[12].hessian, old["hessian"])
    assert torch.equal(gpu_batch.positions[2:5], saved.positions[2:5])
    for state in gpu.bfgs_states.values():
        assert state.device == gpu_batch.positions.device


@pytest.mark.parametrize("variable", [False, True])
def test_hip_single_restart_promotes_with_paired_geometry(variable):
    check_single_restart_promotion(variable, device="cuda", backend="hip")
