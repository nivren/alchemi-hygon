# SPDX-License-Identifier: Apache-2.0
"""Bounded variable-cell FIRE/FIRE2 device parity and public analytic dynamics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell
from nvalchemi.dynamics._ops.fire import fire2_step_coord_cell
from nvalchemi.dynamics._ops.fire_cell import fire_cell_post, fire_cell_pre
from nvalchemi.models.base import BaseModelMixin, ModelConfig


class CellHarmonic(BaseModelMixin):
    """Exact E(H), dE/dH and Cauchy stress for a row-cell analytic fixture."""

    def __init__(self):
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}), supports_pbc=True
        )

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        return data

    def __call__(self, batch):
        h = batch.cell
        gradient = h - 2 * torch.eye(3, dtype=h.dtype, device=h.device)
        return {
            "energy": 0.5 * gradient.square().sum((-1, -2)),
            "forces": torch.zeros_like(batch.positions),
            "stress": h.mT @ gradient / torch.linalg.det(h).abs()[:, None, None],
        }


def single_step(dtype, device, algorithm, branch):
    idx = torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.int32)
    h = torch.tensor(
        [
            [[2.0, 0, 0], [0.4, 2.5, 0], [-0.2, 0.3, 3.0]],
            [[3.0, 0, 0], [-0.3, 2.0, 0], [0.2, -0.4, 2.4]],
        ],
        dtype=dtype,
        device=device,
    )
    frac = torch.arange(15, device=device, dtype=dtype).reshape(5, 3) / 17
    r = torch.bmm(frac[:, None], h[idx.long()]).squeeze(1)
    f = torch.full_like(r, 0.2)
    fh = torch.full_like(h, 0.3)
    if algorithm == "FIRE2":
        fh = torch.tril(fh)
    v, hv = 0.03 * f, 0.02 * fh
    if branch == "negative":
        v, hv = -10 * f, -10 * fh
    elif branch == "zero":
        f.zero_()
        fh.zero_()
        v.zero_()
        hv.zero_()
    elif branch == "cell_only":
        f.zero_()
        v.zero_()
    a = torch.tensor([0.09, 0.04], dtype=dtype, device=device)
    dt = torch.tensor([0.04, 0.06], dtype=dtype, device=device)
    n = torch.tensor([2, 5], dtype=torch.int32, device=device)
    before_r, before_h = r.clone(), h.clone()
    scratch = [torch.empty_like(dt) for _ in range(3)]
    if algorithm == "FIRE2":
        fire2_step_coord_cell(
            r,
            v,
            f,
            h,
            hv,
            fh,
            idx,
            a,
            dt,
            n,
            vf=scratch[0],
            v_sumsq=scratch[1],
            f_sumsq=scratch[2],
            delaystep=2,
            maxstep=0.0002,
            backend="torch_reference",
        )
    else:
        mass = torch.arange(1, 6, dtype=dtype, device=device)
        fire_cell_pre(
            r, v, f, mass, h, hv, fh, dt, idx, maxstep=0.0002, backend="torch_reference"
        )
        fire_cell_post(
            v,
            f,
            mass,
            hv,
            fh,
            idx,
            a,
            dt,
            n,
            torch.full_like(a, 0.1),
            torch.full_like(a, 0.99),
            torch.full_like(a, 0.001),
            torch.full_like(a, 0.2),
            torch.full_like(n, 2),
            torch.full_like(a, 0.5),
            torch.full_like(a, 1.1),
            vf=scratch[0],
            vv=scratch[1],
            ff=scratch[2],
            backend="torch_reference",
        )
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    assert float((r - before_r).norm(dim=-1).max()) <= 0.0002 + tolerance
    if branch == "cell_only":
        after_frac = torch.linalg.solve(h[idx.long()].mT, r[..., None]).squeeze(-1)
        torch.testing.assert_close(after_frac, frac, rtol=tolerance, atol=tolerance)
    if branch == "negative":
        assert not torch.count_nonzero(v) and not torch.count_nonzero(hv)
    if branch == "zero":
        assert torch.equal(r, before_r) and torch.equal(h, before_h)
    result = dict(
        r=r,
        v=v,
        h=h,
        hv=hv,
        alpha=a,
        dt=dt,
        count=n,
        power=scratch[0],
        vv=scratch[1],
        ff=scratch[2],
    )
    assert all(
        t.device == device and (not t.is_floating_point() or t.dtype == dtype)
        for t in result.values()
    )
    # Evidence transfer occurs only after all operation calls/checks.
    return {k: t.cpu() for k, t in result.items()}


def public_run(dtype, device, cls):
    systems = []
    for i, count in enumerate((2, 3)):
        h = torch.tensor(
            [[2.2, 0, 0], [0.1, 2.1, 0], [0.03, -0.02, 2.3]], dtype=dtype, device=device
        ) * (1 + 0.05 * i)
        frac = (
            torch.arange(count * 3, dtype=dtype, device=device).reshape(count, 3) / 11
        )
        r = frac @ h
        systems.append(
            AtomicData(
                positions=r,
                velocities=torch.zeros_like(r),
                forces=torch.zeros_like(r),
                atomic_numbers=torch.ones(count, dtype=torch.long, device=device),
                atomic_masses=torch.ones(count, dtype=dtype, device=device),
                cell=h[None],
                pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
                energy=torch.zeros(1, 1, dtype=dtype, device=device),
                stress=torch.zeros(1, 3, 3, dtype=dtype, device=device),
            )
        )
    b = Batch.from_data_list(systems)
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d.compute(b)
    initial = b.energy.clone()
    frac = torch.linalg.solve(
        b.cell[b.batch_idx.long()].mT, b.positions[..., None]
    ).squeeze(-1)
    d.run(b, n_steps=25)
    assert torch.all(b.energy < initial)
    error = (
        (
            torch.linalg.solve(
                b.cell[b.batch_idx.long()].mT, b.positions[..., None]
            ).squeeze(-1)
            - frac
        )
        .abs()
        .max()
    )
    assert float(error) < (2e-5 if dtype == torch.float32 else 2e-11)
    selected = d._backend_selection if cls is FIRE2VariableCell else d._fire_selection
    return {
        "positions": b.positions.cpu(),
        "cell": b.cell.cpu(),
        "energy": b.energy.cpu(),
    }, {
        "optimizer": cls.__name__,
        "dtype": str(dtype),
        "steps": d.step_count,
        "initial_energy": initial.flatten().tolist(),
        "final_energy": b.energy.flatten().tolist(),
        "fractional_max_abs_error": float(error),
        "selection": selected.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HCU unavailable; no CPU fallback")
    # cuda:0 is explicit in the recorded comparison, avoid unindexed-device equality.
    cases, public = [], []
    for dtype in (torch.float64, torch.float32):
        tolerance = 3e-5 if dtype == torch.float32 else 2e-12
        for algorithm in ("FIRE2", "FIRE"):
            for branch in ("positive", "negative", "zero", "cell_only"):
                expected = single_step(dtype, torch.device("cpu"), algorithm, branch)
                actual = single_step(dtype, device, algorithm, branch)
                error = 0.0
                for key, value in actual.items():
                    torch.testing.assert_close(
                        value, expected[key], rtol=tolerance, atol=tolerance
                    )
                    if value.is_floating_point():
                        error = max(error, float((value - expected[key]).abs().max()))
                cases.append(
                    {
                        "algorithm": algorithm,
                        "branch": branch,
                        "dtype": str(dtype),
                        "cpu_parity_max_abs_error": error,
                    }
                )
        for cls in (FIRE2VariableCell, FIREVariableCell):
            expected, _ = public_run(dtype, torch.device("cpu"), cls)
            actual, info = public_run(dtype, device, cls)
            for key, value in actual.items():
                torch.testing.assert_close(
                    value, expected[key], rtol=tolerance, atol=tolerance
                )
            public.append(info)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(device),
        "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "physical_visible_devices": os.getenv("HIP_VISIBLE_DEVICES"),
        "status": "passed",
        "single_step": cases,
        "public_analytic": public,
        "scope": "synthetic variable-cell optimizer only; no MPA-0/HCU target relaxation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "single_step_cases": len(cases),
                "public_cases": len(public),
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
