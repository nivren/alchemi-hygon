#!/usr/bin/env python3
"""Run a small periodic Torch-reference LJ velocity-Verlet trajectory.

This is deliberately an isolated G1 probe.  It exercises the framework
Batch/NeighborListHook/LJ wrapper chain, but does not claim that the upstream
Warp-backed ``nvalchemi.dynamics.NVE`` public class is portable yet.
"""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.models.lj import LennardJonesModelWrapper


def _wrap_periodic(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap only the periodic x coordinate using the row-vector cell contract."""
    fractional = positions @ torch.linalg.inv(cell)
    wrapped = fractional.clone()
    wrapped[:, 0] = wrapped[:, 0] - torch.floor(wrapped[:, 0])
    return wrapped @ cell


def _evaluate(
    positions: torch.Tensor,
    cell: torch.Tensor,
    model: LennardJonesModelWrapper,
    hook: NeighborListHook,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.ones(positions.shape[0], dtype=torch.long, device=positions.device),
        cell=cell.unsqueeze(0),
        pbc=torch.tensor([[True, False, False]], device=positions.device),
    )
    batch = Batch.from_data_list([data])
    NeighborListHook.__call__.__wrapped__(hook, HookContext(batch=batch), hook.stage)
    output = model(batch)
    return output["energy"].sum(), output["forces"]


def run(device: torch.device, steps: int, dt: float) -> dict[str, object]:
    dtype = torch.float64
    cell = torch.diag(torch.tensor([3.0, 10.0, 10.0], dtype=dtype, device=device))
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype, device=device
    )
    # A common translation crosses the x boundary while leaving the relative
    # pair dynamics unchanged.  The small relative displacement gives a
    # nonzero force and makes the energy test meaningful.
    velocities = torch.tensor(
        [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=dtype, device=device
    )
    masses = torch.ones(2, dtype=dtype, device=device)
    model = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=1.0,
        cutoff=1.5,
        backend="torch_reference",
    )
    hook = NeighborListHook(
        NeighborConfig(cutoff=1.5, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
    )

    potential, forces = _evaluate(positions, cell, model, hook)
    initial_total = potential + 0.5 * (masses[:, None] * velocities.square()).sum()
    sampled: list[dict[str, float | int]] = []
    wrapped_steps: list[int] = []
    max_drift = 0.0

    for step in range(steps + 1):
        kinetic = 0.5 * (masses[:, None] * velocities.square()).sum()
        total = potential + kinetic
        max_drift = max(max_drift, float((total - initial_total).abs().item()))
        if step % max(steps // 4, 1) == 0 or step == steps:
            sampled.append(
                {
                    "step": step,
                    "potential": float(potential.item()),
                    "kinetic": float(kinetic.item()),
                    "total": float(total.item()),
                    "max_drift": float((total - initial_total).abs().item()),
                }
            )
        if step == steps:
            break

        velocities_half = velocities + 0.5 * dt * forces / masses[:, None]
        unwrapped = positions + dt * velocities_half
        if bool(torch.any(unwrapped[:, 0] < 0.0)) or bool(
            torch.any(unwrapped[:, 0] >= cell[0, 0])
        ):
            wrapped_steps.append(step + 1)
        positions = _wrap_periodic(unwrapped, cell)
        velocities = velocities_half
        potential, forces = _evaluate(positions, cell, model, hook)
        velocities = velocities + 0.5 * dt * forces / masses[:, None]

    torch.cuda.synchronize() if device.type == "cuda" else None
    final_kinetic = 0.5 * (masses[:, None] * velocities.square()).sum()
    final_total = potential + final_kinetic
    return {
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "backend": model.backend,
        "steps": steps,
        "dt": dt,
        "initial_total": float(initial_total.item()),
        "final_total": float(final_total.item()),
        "max_drift": max_drift,
        "wrapped_steps": wrapped_steps,
        "total_force": forces.sum(dim=0).tolist(),
        "trajectory": sampled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    args = parser.parse_args()
    if args.steps <= 0 or args.dt <= 0:
        raise ValueError("steps and dt must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but unavailable")
    result = run(device, args.steps, args.dt)
    if result["max_drift"] > 1.0e-8:
        raise RuntimeError(f"NVE energy drift exceeded tolerance: {result['max_drift']}")
    if not result["wrapped_steps"]:
        raise RuntimeError("trajectory did not cross the periodic boundary")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
