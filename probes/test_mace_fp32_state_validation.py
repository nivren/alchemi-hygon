# SPDX-License-Identifier: Apache-2.0
"""CPU tests for separating MACE output and optimizer-state tolerances."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROBES = Path(__file__).resolve().parent
sys.path.insert(0, str(PROBES))

from hip_real_batch_validation import state_check  # noqa: E402


def _batch(force_x: float = 0.07, position_x: float = 0.25):
    dtype = torch.float32
    return SimpleNamespace(
        positions=torch.tensor([[position_x, 0, 0], [1, 0, 0]], dtype=dtype),
        velocities=torch.zeros((2, 3), dtype=dtype),
        cell=torch.eye(3, dtype=dtype).unsqueeze(0) * 5,
        forces=torch.tensor([[force_x, 0.1, 0], [0.2, 0, 0]], dtype=dtype),
        energy=torch.tensor([[1.0]], dtype=dtype),
        stress=torch.eye(3, dtype=dtype).unsqueeze(0) * 0.01,
        batch_ptr=torch.tensor([0, 2]),
        source_index=torch.tensor([[7]]),
        batch_idx=torch.tensor([0, 0]),
        neighbor_list=torch.tensor([[0, 1], [1, 0]]),
        neighbor_list_shifts=torch.zeros((2, 3), dtype=torch.int32),
        num_nodes=2,
    )


def _dynamics(state=None):
    state = torch.tensor(0.02) if state is None else state

    class OptimizerState:
        def __init__(self, dt):
            self.dt = dt

        def __iter__(self):
            return iter((("dt", self.dt),))

    return SimpleNamespace(
        _mutable_fields=("positions", "velocities", "cell", "forces", "energy", "stress"),
        _state=OptimizerState(state),
        step_count=3,
    )


def test_force_cache_uses_physical_tolerance_after_state_update():
    """A valid FP32 force difference is not checked again as a generic state field."""
    actual, expected = _batch(force_x=0.07004), _batch()
    checked = state_check(actual, expected, _dynamics(), _dynamics())
    assert checked["max_abs_errors"]["forces"] == pytest.approx(4e-5, abs=1e-7)
    assert "forces" not in checked["state_max_abs_errors"]
    assert checked["state_max_abs_errors"]["positions"] == 0


def test_force_cache_still_obeys_physical_tolerance():
    """Separating tolerances preserves the existing force-unit acceptance limit."""
    actual, expected = _batch(force_x=0.071), _batch()
    with pytest.raises(AssertionError):
        state_check(actual, expected, _dynamics(), _dynamics())


def test_geometry_and_private_optimizer_state_keep_state_tolerance():
    """Position and private control state still use the original FP32 state tolerance."""
    actual, expected = _batch(position_x=0.2501), _batch()
    with pytest.raises(AssertionError, match="positions"):
        state_check(actual, expected, _dynamics(), _dynamics())

    actual, expected = _batch(), _batch()
    dynamics, reference = _dynamics(torch.tensor(0.021)), _dynamics(torch.tensor(0.02))
    with pytest.raises(AssertionError, match="state.dt"):
        state_check(actual, expected, dynamics, reference)
