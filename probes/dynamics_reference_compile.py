#!/usr/bin/env python3
"""Probe torch.compile boundaries for the Warp-free dynamics reference path."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import torch

from nvalchemi._dynamics_reference.fire import fire2_step_coord
from nvalchemi._dynamics_reference.kinetics import (
    kinetic_energy_per_graph,
    temperature_per_graph,
)


def _inputs(device: torch.device) -> dict[str, tuple[torch.Tensor, ...]]:
    positions = torch.zeros(4, 3, device=device)
    velocities = torch.ones(4, 3, device=device)
    forces = torch.ones(4, 3, device=device)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
    alpha = torch.full((2,), 0.09, device=device)
    dt = torch.full((2,), 0.01, device=device)
    nsteps_inc = torch.zeros(2, dtype=torch.int32, device=device)
    masses = torch.ones(4, device=device)
    atoms_per_graph = torch.tensor([2, 2], device=device)
    return {
        "fire2": (positions, velocities, forces, batch_idx, alpha, dt, nsteps_inc),
        "kinetic": (velocities, masses, batch_idx),
        "temperature": (velocities, masses, batch_idx, atoms_per_graph),
    }


def _fire2_fn(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    fire2_step_coord(
        positions,
        velocities,
        forces,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        delaystep=2,
    )
    return positions, velocities, alpha, dt, nsteps_inc


def _kinetic_fn(
    velocities: torch.Tensor, masses: torch.Tensor, batch_idx: torch.Tensor
) -> torch.Tensor:
    return kinetic_energy_per_graph(velocities, masses, batch_idx, 2)


def _temperature_fn(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    atoms_per_graph: torch.Tensor,
) -> torch.Tensor:
    return temperature_per_graph(
        velocities, masses, batch_idx, 2, atoms_per_graph
    )


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0][:240]


def _explain(fn: Callable[..., Any], args: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    try:
        report = torch._dynamo.explain(fn)(*args)
        reasons = [
            _first_line(getattr(reason, "reason", reason))
            for reason in report.break_reasons
        ]
        return {
            "status": "ran_with_graph_breaks" if report.graph_break_count else "ran",
            "graph_count": report.graph_count,
            "graph_break_count": report.graph_break_count,
            "break_reasons": reasons,
        }
    except Exception as exc:  # pragma: no cover - probe reports environment behavior
        return {"status": "failed", "error": type(exc).__name__, "message": _first_line(exc)}


def _fullgraph(fn: Callable[..., Any], args: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    try:
        torch.compile(fn, backend="eager", fullgraph=True)(*args)
    except Exception as exc:  # pragma: no cover - expected for known limitations
        return {"status": "failed", "error": type(exc).__name__, "message": _first_line(exc)}
    return {"status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cpu":
        raise SystemExit("This compile-boundary probe is currently CPU-only.")

    functions: dict[str, Callable[..., Any]] = {
        "fire2": _fire2_fn,
        "kinetic": _kinetic_fn,
        "temperature": _temperature_fn,
    }
    results: dict[str, Any] = {}
    for name, fn in functions.items():
        results[name] = {
            "default_fullgraph_false": _explain(fn, _inputs(device)[name]),
            "fullgraph_true": _fullgraph(fn, _inputs(device)[name]),
        }
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "device": str(device),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
