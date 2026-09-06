"""Profile cold-start and steady-state stages of the Torch-reference MACE path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(device: torch.device, function):
    _sync(device)
    start = time.perf_counter()
    result = function()
    _sync(device)
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cif", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steady-steps", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.steady_steps < 1:
        raise ValueError("warmup-steps must be non-negative and steady-steps must be positive")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")

    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    cifs = [path.expanduser().resolve() for path in args.cif]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if any(not path.is_file() for path in cifs):
        raise FileNotFoundError(cifs)
    structures = [read(cif) for cif in cifs]
    if not all(bool(atoms.pbc.all()) for atoms in structures):
        raise ValueError("profile requires three-dimensional PBC structures")

    process_start = time.perf_counter()

    def mark(stage: str, **extra: object) -> None:
        print(
            json.dumps(
                {"stage": stage, "elapsed_s": time.perf_counter() - process_start, **extra},
                sort_keys=True,
            ),
            flush=True,
        )

    model_load_s, wrapper = _timed(
        device,
        lambda: MACEWrapper.from_checkpoint(
            str(checkpoint), device=device, dtype=torch.float32
        ),
    )
    wrapper.eval()
    mark(
        "model_loaded",
        duration_s=model_load_s,
        device=(torch.cuda.get_device_name() if args.device == "cuda" else "cpu"),
        checkpoint=str(checkpoint),
    )

    batch_build_s, batch = _timed(
        device,
        lambda: Batch.from_data_list(
            [AtomicData.from_atoms(atoms) for atoms in structures]
        ).to(device),
    )
    mark(
        "batch_built",
        duration_s=batch_build_s,
        num_systems=batch.num_graphs,
        num_nodes=batch.num_nodes,
        atom_counts=[len(atoms) for atoms in structures],
        batch_ptr=batch.batch_ptr.cpu().tolist(),
    )

    config = wrapper.model_config.neighbor_config

    def build_neighbors() -> None:
        compute_neighbors(batch, config=config, backend="torch_reference")

    def run_model():
        adapt_start = time.perf_counter()
        model_inputs = wrapper.adapt_input(batch)
        _sync(device)
        adapt_s = time.perf_counter() - adapt_start
        forward_start = time.perf_counter()
        raw_output = wrapper.model.forward(
            model_inputs,
            compute_force=True,
            compute_stress=False,
            compute_displacement=False,
            training=wrapper.training,
        )
        output = wrapper.adapt_output(raw_output, batch)
        _sync(device)
        forward_s = time.perf_counter() - forward_start
        return adapt_s, forward_s, output

    neighbor_cold_s, _ = _timed(device, build_neighbors)
    mark("neighbors_cold", duration_s=neighbor_cold_s, edges=int(batch.neighbor_list.shape[0]))
    model_cold_start = time.perf_counter()
    adapt_cold_s, forward_cold_s, cold_output = run_model()
    model_cold_s = time.perf_counter() - model_cold_start
    mark(
        "model_cold",
        duration_s=model_cold_s,
        input_adapt_s=adapt_cold_s,
        forward_s=forward_cold_s,
        energy=cold_output["energy"].detach().cpu().tolist(),
    )

    for warmup in range(args.warmup_steps):
        neighbor_s, _ = _timed(device, build_neighbors)
        adapt_s, forward_s, _ = run_model()
        mark(
            "warmup_step",
            step=warmup + 1,
            neighbor_s=neighbor_s,
            input_adapt_s=adapt_s,
            forward_s=forward_s,
        )

    steady = []
    for step in range(args.steady_steps):
        neighbor_s, _ = _timed(device, build_neighbors)
        adapt_s, forward_s, output = run_model()
        record = {
            "step": step + 1,
            "neighbor_s": neighbor_s,
            "input_adapt_s": adapt_s,
            "forward_s": forward_s,
            "energy": output["energy"].detach().cpu().tolist(),
        }
        steady.append(record)
        mark("steady_step", **record)

    mark(
        "summary",
        device=args.device,
        backend="torch_reference",
        num_systems=batch.num_graphs,
        num_nodes=batch.num_nodes,
        edges=int(batch.neighbor_list.shape[0]),
        cold={
            "model_load_s": model_load_s,
            "batch_build_s": batch_build_s,
            "neighbors_s": neighbor_cold_s,
            "model_s": model_cold_s,
            "input_adapt_s": adapt_cold_s,
            "forward_s": forward_cold_s,
        },
        steady_mean_s={
            "neighbor_s": mean(item["neighbor_s"] for item in steady),
            "input_adapt_s": mean(item["input_adapt_s"] for item in steady),
            "forward_s": mean(item["forward_s"] for item in steady),
            "total_s": mean(
                item["neighbor_s"] + item["input_adapt_s"] + item["forward_s"]
                for item in steady
            ),
        },
        steady_steps=steady,
        scope="MACE wrapper Torch-reference cold/steady stage timing",
    )


if __name__ == "__main__":
    main()
