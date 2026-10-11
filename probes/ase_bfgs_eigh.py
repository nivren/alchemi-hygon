# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Isolated FP64 eigensolver assessment; no product backend or Torch patch.

The external torch_rocsolver_eigh strategy is evaluated through DTK's C ABI
using ctypes, independent of its Python 3.10 extension. Input layout, current
stream and solver info are checked. This probe is not a production operator.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch


class NativeSolver:
    """Own a single rocBLAS handle bound to the explicitly selected device."""

    def __init__(self, root, device):
        self.library = ctypes.CDLL(str(root / "lib/librocsolver.so"))
        self.blas = ctypes.CDLL(str(root / "lib/librocblas.so"))
        self.handle = ctypes.c_void_p()
        self.device = device
        self.check(self.blas.rocblas_create_handle(ctypes.byref(self.handle)))
        # All integer widths follow the installed rocSOLVER header, not a guess.
        ptr, i32, i64, f64 = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_double,
        )
        self.library.rocsolver_dsyevj_strided_batched.argtypes = [
            ptr,
            i32,
            i32,
            i32,
            i32,
            ptr,
            i32,
            i64,
            f64,
            ptr,
            i32,
            ptr,
            ptr,
            i64,
            ptr,
            i32,
        ]
        self.library.rocsolver_dsyevd_strided_batched.argtypes = [
            ptr,
            i32,
            i32,
            i32,
            ptr,
            i32,
            i64,
            ptr,
            i64,
            ptr,
            i64,
            ptr,
            i32,
        ]
        self.blas.rocblas_set_stream.argtypes = [ptr, ptr]
        self.blas.rocblas_destroy_handle.argtypes = [ptr]
        self.diagnostics = {}

    @staticmethod
    def check(status):
        if status != 0:
            raise RuntimeError(f"rocBLAS/rocSOLVER API status {status}")

    def close(self):
        self.check(self.blas.rocblas_destroy_handle(self.handle))

    def solve(self, a, algorithm="hybrid"):
        if (
            a.dtype != torch.float64
            or a.device != self.device
            or a.requires_grad
            or not a.is_contiguous()
        ):
            raise ValueError(
                "probe only accepts contiguous, inference FP64 on the selected HCU"
            )
        n, count = a.shape[-1], a.numel() // a.shape[-1] ** 2
        if a.shape[-2] != n:
            raise ValueError("matrix must be square")
        method = (
            "jacobi"
            if algorithm == "hybrid" and n <= 200
            else ("dc" if algorithm == "hybrid" else algorithm)
        )
        stream = torch.cuda.current_stream(self.device)
        self.check(
            self.blas.rocblas_set_stream(
                self.handle, ctypes.c_void_p(stream.cuda_stream)
            )
        )
        work = a.clone(memory_format=torch.contiguous_format)
        w = torch.empty(a.shape[:-1], dtype=a.dtype, device=a.device)
        info = torch.zeros(count, dtype=torch.int32, device=a.device)

        def p(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        # Installed header: evect_original=211, fill_upper=121 (row L == column U).
        if method == "jacobi":
            residual = torch.empty(count, dtype=a.dtype, device=a.device)
            sweeps = torch.zeros(count, dtype=torch.int32, device=a.device)
            status = self.library.rocsolver_dsyevj_strided_batched(
                self.handle,
                252,
                211,
                121,
                n,
                p(work),
                n,
                n * n,
                1e-15,
                p(residual),
                100,
                p(sweeps),
                p(w),
                n,
                p(info),
                count,
            )
            self.diagnostics = {"info": info, "residual": residual, "sweeps": sweeps}
        else:
            scratch = torch.empty_like(w)
            status = self.library.rocsolver_dsyevd_strided_batched(
                self.handle,
                211,
                121,
                n,
                p(work),
                n,
                n * n,
                p(w),
                n,
                p(scratch),
                n,
                p(info),
                count,
            )
            self.diagnostics = {"info": info}
        self.check(status)
        if bool((info != 0).any()):
            raise RuntimeError(f"rocSOLVER {method} solver info: {info.tolist()}")
        return w, work.transpose(-1, -2).contiguous()


def timing(fn, a, warmup, repeats):
    start = time.perf_counter()
    fn(a)
    torch.cuda.synchronize(a.device)
    cold = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        fn(a)
    torch.cuda.synchronize(a.device)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(a)
        torch.cuda.synchronize(a.device)
        values.append((time.perf_counter() - start) * 1000)
    return {
        "first_ms": cold,
        "samples_ms": values,
        "median_ms": statistics.median(values),
    }


def run(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    native = NativeSolver(args.dtk, device)
    cases = []
    try:
        for n in args.sizes:
            for count in args.batches:
                generator = torch.Generator().manual_seed(args.seed + n + count)
                cpu = torch.randn(count, n, n, dtype=torch.float64, generator=generator)
                cpu = (
                    cpu + cpu.mT
                ) * 0.5  # Indefinite spectra exercise ASE's abs eigenvalue semantics.
                a = cpu.to(device)
                reference_w, _ = torch.linalg.eigh(cpu, UPLO="L")
                if args.trace_only:
                    fn = (
                        torch.linalg.eigh
                        if args.solver == "torch"
                        else lambda x: native.solve(x, args.solver)
                    )
                    torch.cuda.synchronize(device)
                    with torch.profiler.record_function(
                        f"eigh.{args.solver}.n{n}.b{count}"
                    ):
                        w, v = fn(a)
                        torch.cuda.synchronize(device)
                    print(
                        f"trace n={n} b={count} solver={args.solver} passed", flush=True
                    )
                    continue
                torch_time = timing(torch.linalg.eigh, a, args.warmup, args.repeats)
                native_time = timing(native.solve, a, args.warmup, args.repeats)
                w, v = native.solve(a)
                normalized = ((v * w.unsqueeze(-2)) @ v.mT - a).norm(
                    dim=(-1, -2)
                ) / a.norm(dim=(-1, -2))
                ortho = (
                    (v.mT @ v - torch.eye(n, dtype=a.dtype, device=device)).abs().amax()
                )
                delta = (w.cpu() - reference_w).abs().max().item()
                residue = normalized.max().item()
                orthogonality = ortho.item()
                if delta > 2e-10 or residue > 1e-11 or orthogonality > 1e-11:
                    raise AssertionError((n, count, delta, residue, orthogonality))
                if not torch.equal(a.cpu(), cpu):
                    raise AssertionError("native candidate mutated its input")
                entry = {
                    "n": n,
                    "batch": count,
                    "torch": torch_time,
                    "native": native_time,
                    "speedup": torch_time["median_ms"] / native_time["median_ms"],
                    "eig_abs_error": delta,
                    "reconstruction_relative": residue,
                    "orthogonality_abs": orthogonality,
                    "solver": "jacobi" if n <= 200 else "dc",
                    "diagnostics": {
                        k: t.cpu().tolist() for k, t in native.diagnostics.items()
                    },
                }
                cases.append(entry)
                print(
                    json.dumps(
                        {
                            k: entry[k]
                            for k in (
                                "n",
                                "batch",
                                "speedup",
                                "reconstruction_relative",
                            )
                        }
                    ),
                    flush=True,
                )
                args.output.write_text(
                    json.dumps({"status": "running", "cases": cases}, indent=2) + "\n"
                )
    finally:
        native.close()
    source = args.reference / "csrc/rocsolver_batched.cpp"
    return {
        "status": "passed",
        "scope": "isolated ctypes C-ABI strategy probe; no product backend",
        "device_name": torch.cuda.get_device_name(device),
        "device": str(device),
        "visible_devices": __import__("os").environ.get("HIP_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_hcu": getattr(torch.version, "__hcu_version__", None),
        "torch_git": torch.version.git_version,
        "hip": torch.version.hip,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": "float64",
        "dtk": str(args.dtk),
        "reference_cpp_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[138, 199, 200, 201, 276, 552]
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtk", type=Path, default=Path("/opt/dtk-26.04"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/home/wangleping/codes/agy-eigh-test/torch_rocsolver_eigh"),
    )
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument(
        "--solver", choices=("torch", "hybrid", "jacobi", "dc"), default="torch"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
