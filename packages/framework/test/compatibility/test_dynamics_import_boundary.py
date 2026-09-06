"""Import-boundary tests for the lazy dynamics namespace."""

from __future__ import annotations

import subprocess
import sys


def test_dynamics_namespace_and_stage_are_warp_free() -> None:
    code = """
import sys
import nvalchemi.dynamics as dynamics
assert 'warp' not in sys.modules
from nvalchemi.dynamics import DynamicsStage
assert DynamicsStage.AFTER_STEP.name == 'AFTER_STEP'
assert 'warp' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_public_nve_fire_wrappers_and_kinetic_hook_select_reference_without_warp() -> None:
    code = """
import sys
import torch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics._ops.fire import fire2_step_coord
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph
assert 'warp' not in sys.modules
assert NVE.__name__ == 'NVE'
pos = torch.zeros((1, 3), dtype=torch.float64)
vel = torch.ones_like(pos)
force = torch.ones_like(pos)
batch = torch.zeros(1, dtype=torch.int32)
alpha = torch.tensor([0.05], dtype=torch.float64)
dt = torch.tensor([0.04], dtype=torch.float64)
nsteps = torch.tensor([1], dtype=torch.int32)
fire2_step_coord(pos, vel, force, batch, alpha, dt, nsteps, backend='torch_reference')
ke = kinetic_energy_per_graph(vel, torch.ones(1, dtype=torch.float64), batch, 1, backend='torch_reference')
assert ke.shape == (1, 1)
assert 'warp' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_velocity_verlet_public_wrapper_can_select_reference_without_warp() -> None:
    code = """
import sys
import torch
from nvalchemi.dynamics._ops.velocity_verlet import vv_position_update
assert 'warp' not in sys.modules
pos = torch.zeros((1, 3), dtype=torch.float64)
vel = torch.zeros_like(pos)
force = torch.ones_like(pos)
mass = torch.ones(1, dtype=torch.float64)
dt = torch.ones(1, dtype=torch.float64)
batch = torch.zeros(1, dtype=torch.int32)
vv_position_update(pos, vel, force, mass, dt, batch, backend='torch_reference')
assert torch.allclose(pos, torch.full_like(pos, 0.5))
assert 'warp' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
