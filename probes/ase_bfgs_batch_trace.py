# SPDX-License-Identifier: Apache-2.0
"""Isolated complete heterogeneous BFGS numerical update, without CPU oracle math."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from ase_bfgs_reference import DeviceGuard
from nvalchemiops.bfgs_ase import bfgs_ase_batch_step, bfgs_ase_prepare_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch_reference", "hip"), required=True)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path, help="Optional separate Torch profiler Chrome trace"
    )
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    widths = [6, 9, 6] if args.small else [147, 285, 147]
    states = [
        bfgs_ase_prepare_state(d, device=device, system_id=i, backend=args.backend)
        for i, d in enumerate(widths)
    ]
    qs = [
        torch.linspace(-0.4, 0.6, d, dtype=torch.float64, device=device).reshape(-1, 3)
        for d in widths
    ]
    ks = [
        torch.arange(1, d + 1, dtype=torch.float64, device=device).reshape(-1, 3) / d
        + 1
        for d in widths
    ]
    guard = DeviceGuard(device)
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        if args.profile
        else nullcontext()
    )
    with guard, profiler:
        for _ in range(3):
            fs = [-k * q for k, q in zip(ks, qs)]
            bfgs_ase_batch_step(qs, fs, states, backend=args.backend)
    torch.cuda.synchronize()
    profile_summary = None
    if args.profile:
        profiler.export_chrome_trace(str(args.profile))
        profile_summary = {
            item.key: dict(
                calls=item.count,
                cpu_total_us=item.cpu_time_total,
                device_total_us=item.device_time_total,
            )
            for item in profiler.key_averages()
            if item.key.startswith("bfgs.")
            or item.key
            in (
                "aten::_local_scalar_dense",
                "aten::clone",
                "aten::bmm",
                "nvalchemiops::_bfgs_eigh_hip",
            )
        }
    result = dict(
        status="passed",
        backend=args.backend,
        scalar_dofs=widths,
        eigh_group_shapes=[[2, widths[0], widths[0]], [1, widths[1], widths[1]]],
        updates=[s.step_count for s in states],
        dtype=str(qs[0].dtype),
        device=str(device),
        guard_calls=dict(guard.calls),
        profiler=profile_summary,
        scope="complete grouped Ops math; no model/Filter/CPU numerical oracle in traced process",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "guard_calls"}), flush=True
    )


if __name__ == "__main__":
    main()
