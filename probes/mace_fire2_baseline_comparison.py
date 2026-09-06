"""Compare heterogeneous and single-system MACE/FIRE2 reference trajectories.

The probe uses the same 46- and 92-atom periodic CIFs for three independent
fresh batches: ``mixed``, ``single_46`` and ``single_92``.  It is deliberately
eager and fixed-cell; the purpose is to expose Batch boundary or state
isolation errors before a longer HCU relaxation is attempted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


class _TrajectoryStep:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, ctx, stage) -> None:
        ctx.batch.trajectory_step.fill_(ctx.step_count + 1)


class _ProgressLogger:
    stage = DynamicsStage.AFTER_STEP

    def __init__(self, label: str, interval: int, start: float, limit: float) -> None:
        self.label = label
        self.frequency = interval
        self.start = start
        self.limit = limit

    def __call__(self, ctx, stage) -> None:
        fmax = _fmax_per_graph(ctx.batch)
        if not torch.isfinite(fmax).all():
            raise FloatingPointError(f"{self.label}: non-finite fmax")
        status = ctx.batch.status.squeeze(-1)
        active = status < 1
        elapsed = time.monotonic() - self.start
        print(
            json.dumps(
                {
                    "stage": "progress",
                    "case": self.label,
                    "step": ctx.step_count + 1,
                    "elapsed_s": elapsed,
                    "fmax": fmax.detach().cpu().tolist(),
                    "active_count": int(active.sum().item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if elapsed >= self.limit:
            raise TimeoutError(f"{self.label}: exceeded max_wall_seconds={self.limit}")


def _fmax_per_graph(batch: Batch) -> torch.Tensor:
    norms = torch.linalg.vector_norm(batch.forces, dim=-1)
    result = torch.full(
        (batch.num_graphs,), float("-inf"), dtype=norms.dtype, device=batch.device
    )
    result.scatter_reduce_(0, batch.batch_idx.long(), norms, reduce="amax", include_self=False)
    return result


def _make_batch(structures: list[Any], system_ids: list[int], device: torch.device) -> Batch:
    data_list = []
    for atoms in structures:
        data = AtomicData.from_atoms(atoms)
        data.add_node_property("velocities", torch.zeros(len(atoms), 3))
        data.forces = torch.zeros(len(atoms), 3)
        data.energy = torch.zeros(1, 1)
        data_list.append(data)
    batch = Batch.from_data_list(data_list).to(device)
    batch.status = torch.zeros((len(data_list), 1), dtype=torch.long, device=device)
    batch.system_id = torch.tensor(
        [[value] for value in system_ids], dtype=torch.long, device=device
    )
    batch.trajectory_step = torch.zeros(
        (len(data_list), 1), dtype=torch.long, device=device
    )
    return batch


def _cross_system_edges(batch: Batch) -> int:
    if not hasattr(batch, "neighbor_list") or batch.neighbor_list.numel() == 0:
        return 0
    src = batch.neighbor_list[:, 0].long()
    dst = batch.neighbor_list[:, 1].long()
    return int((batch.batch_idx[src] != batch.batch_idx[dst]).sum().item())


def _run_case(
    label: str,
    structures: list[Any],
    system_ids: list[int],
    model: MACEWrapper,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    batch = _make_batch(structures, system_ids, device)
    convergence = ConvergenceHook.from_fmax(
        threshold=args.fmax, source_status=0, target_status=1
    )
    start = time.monotonic()
    progress = _ProgressLogger(label, args.log_every, start, args.max_wall_seconds)
    sink = HostMemory(capacity=args.max_steps * len(structures))
    snapshot = SnapshotHook(sink=sink, frequency=1)
    dynamics = FIRE2(
        model=model,
        dt=args.dt,
        n_steps=args.max_steps,
        convergence_hook=convergence,
        hooks=[convergence, _TrajectoryStep(), snapshot, progress],
        backend="torch_reference",
    )
    dynamics.register_hook(
        NeighborListHook(
            model.model_config.neighbor_config,
            skin=args.skin,
            backend="torch_reference",
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    )
    print(
        json.dumps(
            {
                "stage": "case_start",
                "case": label,
                "atom_counts": [len(atoms) for atoms in structures],
                "batch_ptr": batch.batch_ptr.cpu().tolist(),
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dynamics.run(batch)
    # Align final forces with the returned coordinates after graduated systems
    # have been restored by BaseDynamics.
    dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
    dynamics.compute(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    stored = sink.read()
    step_count = dynamics.step_count
    counts = [len(atoms) for atoms in structures]
    expected_ptr = [0]
    for _ in range(step_count):
        for count in counts:
            expected_ptr.append(expected_ptr[-1] + count)
    if stored.batch_ptr.cpu().tolist() != expected_ptr:
        raise AssertionError(
            f"{label}: unexpected stored batch_ptr {stored.batch_ptr.cpu().tolist()}"
        )
    if not torch.all(stored.batch_idx[1:] >= stored.batch_idx[:-1]):
        raise AssertionError(f"{label}: stored batch_idx is not non-decreasing")
    expected_ids = system_ids * step_count
    if stored.system_id.squeeze(-1).cpu().tolist() != expected_ids:
        raise AssertionError(f"{label}: system IDs changed across trajectory frames")
    expected_steps = sum(([step] * len(structures) for step in range(1, step_count + 1)), [])
    if stored.trajectory_step.squeeze(-1).cpu().tolist() != expected_steps:
        raise AssertionError(f"{label}: trajectory_step does not recover frame boundaries")
    cross_edges = _cross_system_edges(batch)
    if cross_edges:
        raise AssertionError(f"{label}: final neighbor list crossed systems: {cross_edges}")
    fmax = _fmax_per_graph(batch).detach().cpu().tolist()
    status = batch.status.squeeze(-1).cpu().tolist()
    result = {
        "case": label,
        "status": "passed",
        "batch_ptr": batch.batch_ptr.cpu().tolist(),
        "stored_batch_ptr": stored.batch_ptr.cpu().tolist(),
        "system_ids": system_ids,
        "stored_system_ids": stored.system_id.squeeze(-1).cpu().tolist(),
        "stored_trajectory_steps": stored.trajectory_step.squeeze(-1).cpu().tolist(),
        "stored_frames": step_count,
        "neighbor_edges": int(batch.neighbor_list.shape[0]),
        "cross_system_edges": cross_edges,
        "status": status,
        "converged": [value >= 1 for value in status],
        "final_fmax": fmax,
        "step_count": step_count,
        "elapsed_s": time.monotonic() - start,
    }
    print(json.dumps({"stage": "case_summary", **result}, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cif-46", type=Path,
        default=Path("/data/csp_data/perf_46/formal_c1_1_100_z1_46.cif"),
    )
    parser.add_argument(
        "--cif-92", type=Path,
        default=Path("/data/csp_data/perf_v2_sorted/perf_v2_92/formal_c1_2_1000_z1_46.cif"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=90.0)
    parser.add_argument("--fmax-compare-atol", type=float, default=5.0e-3)
    parser.add_argument("--require-converged", action="store_true")
    parser.add_argument(
        "--only-case",
        choices=("mixed", "single_46", "single_92"),
        help="run one case without the mixed/single comparison (useful for bounded runs)",
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.fmax <= 0 or args.dt <= 0 or args.skin < 0:
        raise ValueError("max-steps, fmax, dt must be positive; skin must be non-negative")
    if args.log_every < 1 or args.max_wall_seconds <= 0 or args.fmax_compare_atol < 0:
        raise ValueError("log-every and max-wall-seconds must be positive")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")

    paths = [args.cif_46.expanduser().resolve(), args.cif_92.expanduser().resolve()]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(paths)
    structures = [read(path) for path in paths]
    atom_counts = [len(atoms) for atoms in structures]
    if atom_counts != [46, 92]:
        raise ValueError(f"expected [46, 92] atoms, got {atom_counts}")
    if not all(bool(atoms.pbc.all()) for atoms in structures):
        raise ValueError("baseline comparison requires three-dimensional PBC")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    model = MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=torch.float32)
    model.eval()
    supported = {int(z) for z in model.model.atomic_numbers.detach().cpu().tolist()}
    observed = {int(z) for atoms in structures for z in atoms.numbers.tolist()}
    unknown = sorted(observed - supported)
    if unknown:
        raise ValueError(f"input contains unsupported atomic numbers: {unknown}")

    cases = {
        "mixed": (structures, [46, 92]),
        "single_46": ([structures[0]], [46]),
        "single_92": ([structures[1]], [92]),
    }
    selected_cases = cases if args.only_case is None else {args.only_case: cases[args.only_case]}
    results = {
        label: _run_case(label, case_structures, ids, model, args, device)
        for label, (case_structures, ids) in selected_cases.items()
    }
    if args.only_case is not None:
        all_converged = all(all(item["converged"]) for item in results.values())
        if args.require_converged and not all_converged:
            raise AssertionError(f"{args.only_case} did not converge")
        print(
            json.dumps(
                {
                    "stage": "single_case_summary",
                    "status": "passed",
                    "device": args.device,
                    "backend": "torch_reference",
                    "all_converged": all_converged,
                    "case": args.only_case,
                    "result": results[args.only_case],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    mixed_fmax = dict(zip(results["mixed"]["system_ids"], results["mixed"]["final_fmax"]))
    single_fmax = {
        46: results["single_46"]["final_fmax"][0],
        92: results["single_92"]["final_fmax"][0],
    }
    deltas = {str(key): abs(mixed_fmax[key] - single_fmax[key]) for key in (46, 92)}
    if any(value > args.fmax_compare_atol for value in deltas.values()):
        raise AssertionError(
            f"mixed/single final fmax delta exceeds tolerance {args.fmax_compare_atol}: {deltas}"
        )
    all_converged = all(all(item["converged"]) for item in results.values())
    if args.require_converged and not all_converged:
        raise AssertionError("not all mixed/single systems converged")
    print(
        json.dumps(
            {
                "stage": "comparison_summary",
                "status": "passed",
                "device": args.device,
                "backend": "torch_reference",
                "all_converged": all_converged,
                "mixed_single_final_fmax_delta": deltas,
                "fmax_compare_atol": args.fmax_compare_atol,
                "cases": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
