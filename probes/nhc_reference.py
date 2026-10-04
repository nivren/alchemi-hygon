"""Bounded single-device fixed-cell NHC evidence; no device fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTNoseHoover
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper
from nvalchemiops.torch_nhc import nhc_chain_update_functional, nhc_compute_masses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available() and torch.version.hip, "HIP device required"
    parity = []
    for dtype in (torch.float32, torch.float64):
        v = torch.arange(18, dtype=dtype).reshape(6, 3) * 0.001 + 0.01
        m = torch.arange(6, dtype=dtype) + 1
        idx = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32)
        kt = torch.tensor([0.025, 0.05], dtype=dtype)
        dt = torch.tensor([0.01, 0.02], dtype=dtype)
        tau = torch.tensor([1.0, 2.0], dtype=dtype)
        ndof = torch.tensor([6.0, 12.0], dtype=dtype)
        for length in (1, 3, 8):
            q = nhc_compute_masses(kt, tau, m, idx, length)
            for loops in (3, 5):
                eta = torch.zeros_like(q)
                u = torch.full_like(q, 0.001)
                inputs = (v, m, eta, u, q, kt, dt, ndof, idx)
                expected = nhc_chain_update_functional(*inputs, nloops=loops)
                actual = nhc_chain_update_functional(
                    *(x.to(device) for x in inputs), nloops=loops
                )
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(
                        a.cpu(),
                        b,
                        rtol=3e-5 if dtype == torch.float32 else 1e-11,
                        atol=3e-7 if dtype == torch.float32 else 1e-13,
                    )
                parity.append([str(dtype), length, loops])
    print(json.dumps({"stage": "parity", "cases": parity}), flush=True)
    model = MACEWrapper.from_checkpoint(
        str(args.checkpoint), device=device, dtype=torch.float32
    )
    model.eval()
    systems = []
    for name in ("H2O", "CH4"):
        atoms = molecule(name)
        atoms.set_cell([12.0, 12.0, 12.0])
        atoms.center()
        atoms.pbc = True
        data = AtomicData.from_atoms(atoms)
        data.add_node_property("velocities", torch.full((len(atoms), 3), 0.001))
        data.forces = torch.zeros(len(atoms), 3)
        data.energy = torch.zeros(1, 1)
        systems.append(data)
    b = Batch.from_data_list(systems).to(device)
    cell = b.cell.clone()

    def create():
        d = NVTNoseHoover(
            model,
            dt=0.1,
            temperature=torch.tensor([300.0, 600.0], device=device),
            thermostat_time=10.0,
            backend="torch_reference",
        )
        d.register_hook(
            NeighborListHook(
                model.model_config.neighbor_config,
                skin=0.5,
                backend="torch_reference",
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        )
        return d

    d = create()
    d._call_hooks(DynamicsStage.BEFORE_COMPUTE, b)
    d.compute(b)
    d.run(b, n_steps=2)
    snapshot = b.clone()
    saved = d.state_dict()
    d.run(b, n_steps=1)
    resumed = create()
    resumed.load_state_dict(saved)
    resumed.run(snapshot, n_steps=1)
    for key in ("positions", "velocities", "forces", "energy"):
        torch.testing.assert_close(
            getattr(b, key), getattr(snapshot, key), rtol=3e-5, atol=1e-6
        )
    assert (
        torch.equal(b.cell, cell)
        and torch.isfinite(b.forces).all()
        and torch.isfinite(b.velocities).all()
    )
    initial = d._state.nhc_eta.clone()
    b["status"] = torch.tensor([[0], [1]], device=device)
    d.step(b)
    torch.testing.assert_close(d._state.nhc_eta[1], initial[1], rtol=0, atol=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "stage": "mace_pass",
                "device": str(device),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "name": torch.cuda.get_device_name()
                if device.type == "cuda"
                else "CPU",
                "batch_ptr": b.batch_ptr.cpu().tolist(),
                "steps": d.step_count,
                "restart_steps": resumed.step_count,
                "energy_eV": b.energy.detach().cpu().tolist(),
                "fixed_cell": True,
                "inactive_chain_preserved": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
