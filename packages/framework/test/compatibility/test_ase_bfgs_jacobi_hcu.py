# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit product backend, whole-Batch model and native workflow lifecycle gates."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics import (
    ASECompatibleBFGS,
    ASECompatibleBFGSVariableCell,
    FusedStage,
)
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from test.compatibility.test_ase_bfgs_batch import (
    BatchCell,
    BatchPolynomial,
    Dataset,
    check_single_restart_promotion,
    mixed,
)
from test.compatibility.test_ase_bfgs_cell import assert_snapshot_equal

BACKEND = "hip.bfgs-ase-jacobi-streams-v1"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires allocated HIP device",
)


@pytest.mark.parametrize(
    "variable,options",
    [
        (False, {}),
        (True, {}),
        (True, dict(cell_factor=2.7, scalar_pressure=0.037)),
        (True, dict(mask=[1, 0, 1, 0, 1, 0])),
    ],
)
def test_product_batch_cpu_continuation_and_id_lifecycle(variable, options):
    """Explicit registry dispatch keeps force sharing, masks, history and geometry."""
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    model_cls = BatchCell if variable else BatchPolynomial
    cpu = mixed(variable)
    gpu = cpu.clone().to("cuda")
    reference = cls(model_cls(), fmax=1e-10, **options)
    candidate = cls(model_cls(), backend=BACKEND, fmax=1e-10, **options)
    alternate = torch.cuda.Stream()
    alternate.wait_stream(torch.cuda.current_stream())
    for _ in range(7):
        reference.step(cpu)
        with torch.cuda.stream(alternate):
            candidate.step(gpu)
            done = torch.cuda.Event()
            done.record()
        torch.cuda.current_stream().wait_event(done)
        assert candidate._batch_data.selection.implementation_id == BACKEND
        torch.testing.assert_close(
            gpu.positions.cpu(), cpu.positions, atol=5e-12, rtol=5e-12
        )
        if variable:
            torch.testing.assert_close(gpu.cell.cpu(), cpu.cell, atol=5e-12, rtol=5e-12)
        for identifier in reference.bfgs_states:
            torch.testing.assert_close(
                candidate.bfgs_states[identifier].hessian.cpu(),
                reference.bfgs_states[identifier].hessian,
                atol=6e-10,
                rtol=6e-11,
            )
    assert candidate.model.calls == reference.model.calls == 8
    snapshot = candidate.state_dict()
    paired = gpu.clone()
    resumed = cls(model_cls(), backend=BACKEND, fmax=1e-10, **options)
    resumed.load_state_dict(snapshot)
    candidate.step(gpu)
    resumed.step(paired)
    assert torch.equal(gpu.positions, paired.positions)
    if variable:
        assert torch.equal(gpu.cell, paired.cell)
    old = candidate.bfgs_states[12].state_dict()
    saved = gpu.clone()
    candidate._masked_pre_update(gpu, torch.tensor([True, False, True], device="cuda"))
    assert_snapshot_equal(old, candidate.bfgs_states[12].state_dict())
    assert torch.equal(gpu.positions[2:5], saved.positions[2:5])
    if variable:
        assert torch.equal(gpu.cell[1], saved.cell[1])
    before = {i: s.state_dict() for i, s in candidate.bfgs_states.items()}
    reordered = gpu.index_select(torch.tensor([2, 0, 1], device="cuda"))
    candidate._ensure_state_initialized(reordered)
    for i, s in candidate.bfgs_states.items():
        assert_snapshot_equal(before[i], s.state_dict())
    data = reordered.to_data_list()
    data[2].system_id.fill_(90)
    replaced = Batch.from_data_list(data[:1] + data[2:])
    survivor = candidate.bfgs_states[13]
    candidate._ensure_state_initialized(replaced)
    assert (
        set(candidate.bfgs_states) == {13, 90} and candidate.bfgs_states[13] is survivor
    )
    assert candidate.bfgs_states[90].step_count == 0


@pytest.mark.parametrize("variable", [False, True])
def test_product_single_restart_promotion_and_native_refill(variable):
    """Sampler-driven FusedStage exits/refills keep IDs and shared model calls."""
    check_single_restart_promotion(variable, device="cuda", backend=BACKEND)
    cls = ASECompatibleBFGSVariableCell if variable else ASECompatibleBFGS
    model = BatchCell() if variable else BatchPolynomial()
    d = cls(model, backend=BACKEND, fmax=1e-10, n_steps=3, device_type="cuda")
    sink = HostMemory(capacity=4)
    dataset = Dataset(variable)
    dataset.rows = [row.to("cuda") for row in dataset.rows]
    stage = FusedStage(
        [(0, d)],
        sampler=SizeAwareSampler(dataset, max_atoms=6, max_batch_size=2),
        sinks=[sink],
        refill_frequency=1,
        device_type="cuda",
    )
    assert stage.run(n_steps=20) is None
    assert len(sink) == 4 and sorted(
        sink.read().source_index.flatten().tolist()
    ) == list(range(4))
    assert model.calls == stage.step_count + 2 and not d.bfgs_states
