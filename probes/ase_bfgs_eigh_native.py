# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""E1 product solver numerical/performance and library-internal trace probe."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from nvalchemiops._bfgs_eigh_hip import _eigh, _load_extension, _rocsolver_eigh
from nvalchemiops.bfgs_ase import bfgs_ase_prepare_state, bfgs_ase_step


def measure(fn, a, warmup, repeats):
    """Time complete callable cost including its layout and status checks."""
    start = time.perf_counter()
    fn(a)
    torch.cuda.synchronize(a.device)
    cold = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        fn(a)
    torch.cuda.synchronize(a.device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(a)
        torch.cuda.synchronize(a.device)
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "first_ms": cold,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
    }


def run(args):
    """Use CPU FP64 eigenvalues and device reconstruction as independent gates."""
    torch.cuda.set_device(0)
    start = time.perf_counter()
    extension = _load_extension() if args.solver == "hip" else None
    load_ms = (time.perf_counter() - start) * 1000
    cases = []
    for n in args.sizes:
        for count in args.batches:
            if args.bfgs_steps:
                if n % 3 or count != 1 or not args.trace_only:
                    raise ValueError(
                        "BFGS trace requires dof divisible by 3, B=1 and trace-only"
                    )
                backend = "hip" if args.solver == "hip" else "torch_reference"
                state = bfgs_ase_prepare_state(n, device="cuda", backend=backend)
                q = torch.linspace(
                    -0.4, 0.6, n, dtype=torch.float64, device="cuda"
                ).reshape(-1, 3)
                k = torch.arange(1, n + 1, dtype=q.dtype, device=q.device).reshape_as(q)
                for _ in range(args.bfgs_steps):
                    bfgs_ase_step(q, -(k * q + 0.3 * q**3), state, backend=backend)
                torch.cuda.synchronize()
                print(
                    f"trace BFGS backend={backend} n={n} steps={args.bfgs_steps} passed",
                    flush=True,
                )
                continue
            cpu = torch.randn(
                count,
                n,
                n,
                dtype=torch.float64,
                generator=torch.Generator().manual_seed(24 + n + count),
            )
            cpu = (cpu + cpu.mT) * 0.5
            a = cpu.cuda()
            before = a.clone()
            torch.cuda.synchronize()
            if args.trace_only:
                fn = _eigh if args.solver == "hip" else torch.linalg.eigh
                with torch.profiler.record_function(
                    f"eigh.{args.solver}.n{n}.b{count}"
                ):
                    fn(a)
                    torch.cuda.synchronize()
                print(f"trace solver={args.solver} n={n} B={count} passed", flush=True)
                continue
            native = measure(_eigh, a, args.warmup, args.repeats)
            reference = measure(torch.linalg.eigh, a, args.warmup, args.repeats)
            w, v, info, residual, sweeps = _rocsolver_eigh(a, "L", 100)
            target, _ = torch.linalg.eigh(cpu)
            diff = (w.cpu() - target).abs().max().item()
            reconstruction = ((v * w.unsqueeze(-2)) @ v.mT - a).norm(
                dim=(-1, -2)
            ) / a.norm(dim=(-1, -2))
            orthogonal = (v.mT @ v - torch.eye(n, dtype=a.dtype, device=a.device)).norm(
                dim=(-1, -2)
            )
            rmax, omax = reconstruction.max().item(), orthogonal.max().item()
            torch.testing.assert_close(w.cpu(), target, atol=2e-10, rtol=2e-11)
            if rmax >= 1e-11 or omax >= 1e-11 or not torch.equal(a, before):
                raise AssertionError("solver invariant/input-preservation gate failed")
            case = {
                "n": n,
                "batch": count,
                "algorithm": "jacobi" if n < 200 else "dc",
                "torch_ms": reference,
                "native_ms": native,
                "speedup": reference["median_ms"] / native["median_ms"],
                "max_eigenvalue_abs_error": diff,
                "reconstruction_relative": rmax,
                "orthogonality_norm": omax,
                "info": info.tolist(),
                "jacobi_residual": residual.tolist() if n < 200 else None,
                "jacobi_sweeps": sweeps.tolist() if n < 200 else None,
            }
            cases.append(case)
            print(json.dumps(case), flush=True)
    return {
        "status": "passed",
        "torch": torch.__version__,
        "torch_hcu": getattr(torch.version, "__hcu_version__", None),
        "hip": torch.version.hip,
        "rocsolver": extension.rocsolver_version() if extension else None,
        "device_name": torch.cuda.get_device_name(),
        "dtype": "float64",
        "extension_load_ms": load_ms,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "trace_only": args.trace_only,
        "timing_scope": "complete product eigh with finite/info checks, allocation and output layout; sync wall clock; JIT excluded",
        "cases": cases,
    }


def main():
    """Run a bounded list of sizes/batches and keep machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[138, 199, 200, 201, 276, 552]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--bfgs-steps", type=int, default=0)
    parser.add_argument("--solver", choices=("torch", "hip"), default="hip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes + args.batches + [args.repeats]) <= 0 or args.warmup < 0:
        parser.error("dimensions/batches/repeats positive, warmup nonnegative required")
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
