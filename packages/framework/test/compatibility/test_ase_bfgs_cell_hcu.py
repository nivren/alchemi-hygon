# SPDX-License-Identifier: Apache-2.0
"""Explicit HCU-only M3 restart, stream and complete inactive-state checks."""

import pytest
import torch

from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from test.compatibility.test_ase_bfgs_cell import (
    CellModel,
    assert_snapshot_equal,
    make_batch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not getattr(torch.version, "__hcu_version__", None),
    reason="requires the admitted HCU PyTorch build and an allocated device",
)


@pytest.mark.parametrize(
    "backend", ["torch_reference", "hip", "hip.bfgs-ase-jacobi-streams-v1"]
)
def test_cpu_snapshot_explicit_hcu_restore_preserves_filter_and_continuation(backend):
    cpu_batch = make_batch()
    cpu = ASECompatibleBFGSVariableCell(
        CellModel(), scalar_pressure=0.02, cell_factor=2.7, fmax=1e-10
    )
    cpu.run(cpu_batch, n_steps=2)
    snapshot = cpu.state_dict()
    hcu_batch = cpu_batch.clone().to("cuda")
    hcu = ASECompatibleBFGSVariableCell(CellModel(), backend=backend)
    hcu.load_state_dict(snapshot, device="cuda")
    assert (
        hcu.filter_state.orig_cell.device
        == hcu.bfgs_state.device
        == hcu_batch.positions.device
    )
    for _ in range(4):
        cpu.step(cpu_batch)
        hcu.step(hcu_batch)
        torch.testing.assert_close(
            hcu_batch.positions.cpu(), cpu_batch.positions, atol=3e-12, rtol=3e-12
        )
        torch.testing.assert_close(
            hcu_batch.cell.cpu(), cpu_batch.cell, atol=3e-12, rtol=3e-12
        )
        torch.testing.assert_close(
            hcu.bfgs_state.hessian.cpu(), cpu.bfgs_state.hessian, atol=5e-10, rtol=3e-11
        )
    # Source snapshot stays on CPU with independently owned parameter storage.
    assert snapshot["filter_state"]["orig_cell"].device.type == "cpu"


@pytest.mark.parametrize(
    "backend", ["torch_reference", "hip", "hip.bfgs-ase-jacobi-streams-v1"]
)
def test_variable_cell_updates_and_restore_obey_nondefault_stream(backend):
    # Prepare native library on the default stream, then run actual tensor work
    # and continuation on another stream with an explicit consumer event.
    b = make_batch("cuda")
    default = ASECompatibleBFGSVariableCell(CellModel(), backend=backend, fmax=1e-10)
    default._ensure_state_initialized(b)
    alternate = torch.cuda.Stream()
    alternate.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alternate):
        default.run(b, n_steps=2)
        snapshot = default.state_dict()
        clone = b.clone()
        resumed = ASECompatibleBFGSVariableCell(CellModel(), backend=backend)
        resumed.load_state_dict(snapshot)
        default.step(b)
        resumed.step(clone)
        done = torch.cuda.Event()
        done.record()
    torch.cuda.current_stream().wait_event(done)
    torch.testing.assert_close(b.positions, clone.positions, atol=0, rtol=0)
    torch.testing.assert_close(b.cell, clone.cell, atol=0, rtol=0)
    torch.testing.assert_close(
        default.bfgs_state.hessian, resumed.bfgs_state.hessian, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "backend", ["torch_reference", "hip", "hip.bfgs-ase-jacobi-streams-v1"]
)
@pytest.mark.parametrize("method", ["step", "masked_update"])
def test_hcu_inactive_cell_filter_and_history_are_frozen(backend, method):
    b = make_batch("cuda")
    d = ASECompatibleBFGSVariableCell(CellModel(), backend=backend, fmax=1e-10)
    d.run(b, n_steps=2)
    snapshot, saved = d.state_dict(), b.clone()
    if method == "step":
        b.status = torch.tensor([[d.exit_status]], device="cuda")
        d.step(b)
    else:
        d.masked_update(b, torch.tensor([False], device="cuda"))
    now = d.state_dict()
    now["step_count"] = snapshot["step_count"]
    assert_snapshot_equal(snapshot, now)
    for key in ("positions", "cell", "forces", "energy", "stress"):
        assert torch.equal(getattr(b, key), getattr(saved, key))
