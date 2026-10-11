# SPDX-License-Identifier: Apache-2.0
"""Strict no-clamp diagonal eigensystem and public ASE BFGS regression."""

import argparse
import json
from pathlib import Path

import torch
from nvalchemiops._bfgs_jacobi_hip import _eigh
from nvalchemiops.bfgs_ase import bfgs_ase_prepare_state, bfgs_ase_step

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
backend = "hip.bfgs-ase-jacobi-streams-v1"
diagonal = torch.tensor([2e-18, 1e-18, 2.0], dtype=torch.float64)
h = torch.diag(diagonal).cuda()
w, v = _eigh(h)
diagnostic = dict(
    status="diagnostic", observed=w.tolist(), expected=diagonal.sort().values.tolist()
)
a.output.write_text(json.dumps(diagnostic, indent=2) + "\n")
print(json.dumps(diagnostic), flush=True)
torch.testing.assert_close(w.cpu(), diagonal.sort().values, atol=0, rtol=0)
cpu = torch.zeros(1, 3, dtype=torch.float64)
gpu = cpu.cuda()
force = torch.tensor([[2e-19, 1e-19, 0.0]], dtype=torch.float64)
reference = bfgs_ase_prepare_state(3, system_id=0)
reference.hessian.copy_(torch.diag(diagonal))
state = bfgs_ase_prepare_state(3, system_id=0, device="cuda", backend=backend)
state.hessian.copy_(h)
expected = bfgs_ase_step(cpu, force, reference)
actual = bfgs_ase_step(gpu, force.cuda(), state, backend=backend)
torch.testing.assert_close(gpu.cpu(), cpu, atol=0, rtol=0)
torch.testing.assert_close(actual.raw_step.cpu(), expected.raw_step, atol=0, rtol=0)
result = dict(
    status="passed",
    eigenvalues=w.tolist(),
    raw_step=actual.raw_step.tolist(),
    spectrum_and_public_bfgs_cpu_bitwise=True,
    no_clamp=True,
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
