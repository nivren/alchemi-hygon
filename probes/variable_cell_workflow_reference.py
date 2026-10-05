# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic CPU/HCU workflow parity; no real MACE model here."""

import argparse
import json

import torch

from nvalchemi.data import AtomicData
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell, FusedStage
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.base import NeighborConfig
from variable_cell_fire_reference import CellHarmonic


class Dataset:
    def __init__(self, dtype, device):
        self.dtype, self.device = dtype, device

    def __len__(self):
        return 4

    def get_metadata(self, i):
        return (2, 3, 3, 2)[i], 0

    def __getitem__(self, i):
        n = self.get_metadata(i)[0]
        h = torch.diag(
            torch.tensor(
                [2.12 + i * 0.03, 2.18, 2.2], dtype=self.dtype, device=self.device
            )
        )
        r = torch.arange(n * 3, dtype=self.dtype, device=self.device).reshape(n, 3) / 10
        data = AtomicData(
            positions=r,
            atomic_numbers=torch.ones(n, dtype=torch.long, device=self.device),
            atomic_masses=torch.ones(n, dtype=self.dtype, device=self.device),
            velocities=torch.zeros_like(r),
            forces=torch.zeros_like(r),
            energy=torch.zeros(1, 1, dtype=self.dtype, device=self.device),
            stress=torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device),
            cell=h[None],
            pbc=torch.ones(1, 3, dtype=torch.bool, device=self.device),
        )
        data.add_system_property(
            "source_index", torch.tensor([[i]], device=self.device)
        )
        return data, {}


def run(dtype, device, cls):
    model = CellHarmonic()
    sampler = SizeAwareSampler(Dataset(dtype, device), max_atoms=6, max_batch_size=2)
    sink = HostMemory(capacity=4)
    convergence = ConvergenceHook(
        criteria=[
            {
                "key": "forces",
                "threshold": 0.01,
                "reduce_op": "norm",
                "reduce_dims": -1,
            },
            {
                "key": "stress",
                "threshold": 1e-4,
                "custom_op": lambda x: x.abs().amax((-1, -2)) <= 1e-4,
            },
        ],
        source_status=0,
        target_status=1,
    )
    seen = set()

    class Check:
        stage = DynamicsStage.BEFORE_PRE_UPDATE
        frequency = 1

        def __call__(self, ctx, stage):
            expected = model(ctx.batch)
            tol = 3e-5 if dtype == torch.float32 else 1e-12
            torch.testing.assert_close(
                ctx.batch.stress, expected["stress"], rtol=tol, atol=tol
            )
            assert ctx.batch.cell.device == device
            seen.update(ctx.batch.source_index.flatten().tolist())
            edges = ctx.batch.neighbor_list.long()
            assert torch.equal(
                ctx.batch.batch_idx[edges[:, 0]], ctx.batch.batch_idx[edges[:, 1]]
            )

    dynamics = cls(
        model=model,
        dt=0.04,
        backend="torch_reference",
        convergence_hook=convergence,
        hooks=[Check()],
        device_type=device.type,
    )
    stage = FusedStage(
        sub_stages=[(0, dynamics)],
        sampler=sampler,
        sinks=[sink],
        refill_frequency=1,
        device_type=device.type,
    )
    stage.register_hook(
        NeighborListHook(
            NeighborConfig(cutoff=1.0),
            skin=0,
            backend="torch_reference",
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    )
    assert stage.run(n_steps=400) is None
    stored = sink.read()
    assert len(sink) == 4 and seen == set(range(4))
    assert sorted(stored.system_id.flatten().tolist()) == list(range(4))
    order = torch.argsort(stored.source_index.flatten())
    result = stored.index_select(order)
    assert result.stress.abs().max() <= 1e-4
    # Test private-state protection after cell motion, independently of graduation.
    b = sampler._dataset[0][0]
    from nvalchemi.data import Batch

    b = Batch.from_data_list([b, sampler._dataset[1][0]])
    d = cls(model=model, dt=0.04, backend="torch_reference", device_type=device.type)
    d.compute(b)
    d.step(b)
    previous = {k: v.clone() for k, v in d._state}
    frozen = {k: getattr(b, k).clone() for k in d._mutable_fields}
    b.status = torch.tensor([[0], [1]], device=device)
    d.step(b)
    assert all(torch.equal(getattr(d._state, k)[1], v[1]) for k, v in previous.items())
    assert all(
        torch.equal(
            getattr(b, k)[2 if k in ("positions", "velocities", "forces") else 1 :],
            v[2 if k in ("positions", "velocities", "forces") else 1 :],
        )
        for k, v in frozen.items()
    )
    return {
        k: getattr(result, k).cpu() for k in ("positions", "cell", "stress", "energy")
    }, stage.step_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.device == "cuda" and (
        not torch.version.hip or not torch.cuda.is_available()
    ):
        raise RuntimeError("HIP HCU unavailable")
    cases = []
    for dtype in (torch.float64, torch.float32):
        for cls in (FIRE2VariableCell, FIREVariableCell):
            reference, cpu_steps = run(dtype, torch.device("cpu"), cls)
            actual, steps = (
                run(dtype, device, cls)
                if args.device == "cuda"
                else (reference, cpu_steps)
            )
            tol = 3e-5 if dtype == torch.float32 else 2e-11
            for key in reference:
                torch.testing.assert_close(
                    actual[key], reference[key], rtol=tol, atol=tol
                )
            assert steps == cpu_steps
            cases.append(
                {
                    "optimizer": cls.__name__,
                    "dtype": str(dtype),
                    "steps": steps,
                    "graduated": 4,
                    "max_abs_stress": float(actual["stress"].abs().max()),
                }
            )
    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "device_name": torch.cuda.get_device_name()
                if args.device == "cuda"
                else "cpu",
                "cases": cases,
                "scope": "synthetic native convergence/refill/neighbor/exit state only",
                "HCU_MPA0": "not_run",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
