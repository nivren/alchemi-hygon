#!/usr/bin/env python3
"""Run the framework NeighborListHook -> Lennard-Jones reference chain."""

from __future__ import annotations

import json

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.lj import LennardJonesModelWrapper


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the framework HCU probe requires a CUDA/HIP device")

    device = torch.device("cuda")
    dtype = torch.float64
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                    device=device,
                    dtype=dtype,
                ),
                atomic_numbers=torch.tensor([1, 1], device=device),
            ),
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                    device=device,
                    dtype=dtype,
                ),
                atomic_numbers=torch.tensor([1, 1], device=device),
            ),
        ]
    )
    model = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
        backend="torch_reference",
    )
    (hook,) = model.make_neighbor_hooks()
    hook(HookContext(batch=batch), hook.stage)
    output = model(batch)
    torch.cuda.synchronize()

    expected_energy = torch.tensor(
        [-0.9833724493736826, -0.8909652875830761],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(
        output["energy"].flatten(), expected_energy, rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        output["forces"].sum(dim=0),
        torch.zeros(3, device=device, dtype=dtype),
        rtol=1e-11,
        atol=1e-11,
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "backend": model.backend,
                "neighbor_matrix": batch.neighbor_matrix.tolist(),
                "num_neighbors": batch.num_neighbors.tolist(),
                "energy": output["energy"].flatten().tolist(),
                "force_norm": torch.linalg.vector_norm(output["forces"]).item(),
                "total_force": output["forces"].sum(dim=0).tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
