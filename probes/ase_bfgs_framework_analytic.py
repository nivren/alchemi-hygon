# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-card analytic Framework/ASE/restart/device validation, no MLIP noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS
from ase_bfgs_reference import DeviceGuard
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGS
from nvalchemi.models.base import BaseModelMixin, ModelConfig


class Polynomial(BaseModelMixin):
    """Device-local diagonal polynomial; force evaluations are reproducible."""

    def __init__(self):
        self.model_config = ModelConfig(outputs={"energy", "forces"})
        self.calls = 0

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        return data

    def __call__(self, batch):
        self.calls += 1
        q = batch.positions
        k = torch.arange(1, q.numel() + 1, dtype=q.dtype, device=q.device).view_as(q)
        return {
            "forces": -(k * q + 0.3 * q**3),
            "energy": (0.5 * k * q**2 + 0.075 * q**4).sum(),
        }


def run(device, backend="torch_reference"):
    cases = []
    for n in (1, 3, 12):
        q = torch.linspace(-0.4, 0.6, n * 3, dtype=torch.float64).reshape(n, 3)
        oracle = BFGS(
            Atoms("H" * n, positions=q.numpy()), logfile=None, alpha=53, maxstep=0.12
        )
        b = Batch.from_data_list(
            [
                AtomicData(
                    positions=q,
                    atomic_numbers=torch.ones(n, dtype=torch.long),
                    forces=torch.zeros_like(q),
                    energy=torch.zeros(1, 1, dtype=torch.float64),
                )
            ]
        ).to(device)
        d = ASECompatibleBFGS(
            Polynomial(), alpha=53, maxstep=0.12, fmax=1e-14, backend=backend
        )
        guard = DeviceGuard(b.positions.device)
        twin = None
        errors = {"positions": 0.0, "hessian": 0.0, "raw_step": 0.0}
        for i in range(8):
            x = oracle.atoms.positions.copy()
            k = np.arange(1, x.size + 1).reshape(x.shape)
            f = -(k * x + 0.3 * x**3)
            raw, lengths = oracle.prepare_step(x.ravel(), -f.ravel())
            raw = raw.copy()
            applied = oracle.determine_step(raw.copy(), lengths)
            oracle.atoms.positions = x + applied.reshape(-1, 3)
            with guard:
                d.step(b)
            for key, actual, target in (
                ("positions", b.positions, oracle.atoms.positions),
                ("hessian", d.bfgs_state.hessian, oracle.H),
                ("raw_step", d.last_step.raw_step, raw.reshape(-1, 3)),
            ):
                observed = actual.cpu().numpy()
                errors[key] = max(errors[key], float(np.abs(observed - target).max()))
                np.testing.assert_allclose(observed, target, atol=2e-10, rtol=2e-11)
            if twin is not None:
                cloned, resumed = twin
                with guard:
                    resumed.step(cloned)
                torch.testing.assert_close(
                    cloned.positions, b.positions, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    resumed.bfgs_state.hessian, d.bfgs_state.hessian, atol=0, rtol=0
                )
            if i == 3:
                resumed = ASECompatibleBFGS(Polynomial(), backend=backend)
                resumed.load_state_dict(d.state_dict())
                twin = b.clone(), resumed
        assert d.bfgs_state.step_count == 8 and d.model.calls == 9
        before = d.bfgs_state.state_dict()
        d.masked_update(b, torch.tensor([False], device=device))
        assert torch.equal(before["hessian"], d.bfgs_state.hessian)
        assert d.bfgs_state.step_count == 8
        cases.append(
            {
                "natoms": n,
                "dof": n * 3,
                "updates": 8,
                "model_evaluations": 9,
                "errors_max_abs": errors,
                "restart": "bitwise equal for 4 continuation updates",
                "device_guard": "passed",
                "observed_ops": len(guard.calls),
            }
        )
    return {
        "status": "passed",
        "device": str(device),
        "backend": backend,
        "implementation": d._backend_selection.implementation_id,
        "torch": torch.__version__,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--backend", choices=("torch_reference", "hip"), default="torch_reference"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(torch.device(args.device), args.backend)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
