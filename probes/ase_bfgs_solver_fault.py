# SPDX-License-Identifier: Apache-2.0
"""Isolate a saved CSP Hessian in the product, archived DC or diagnostic solver."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from nvalchemiops._bfgs_eigh_hip import _eigh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("native", "dc", "jacobi", "torch"), required=True
    )
    parser.add_argument(
        "--source-root", type=Path, help="Archived DC sources; required for --method dc"
    )
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu = torch.load(args.matrix, map_location="cpu", weights_only=True)
    expected, _ = torch.linalg.eigh(cpu)
    a = cpu.cuda()
    if args.method in ("jacobi", "dc"):
        from nvalchemiops._hip_extension_loader import load_hip_extension

        source = (
            args.source_root
            if args.method == "dc"
            else (
                Path(__file__).resolve().parents[1]
                / "packages/ops/nvalchemiops/_native"
            )
        )
        if source is None:
            raise ValueError("--method dc requires archived --source-root")
        diagnostic = args.build_root / f"{args.method}-diagnostic-source"
        diagnostic.mkdir(parents=True, exist_ok=True)
        for name in ("bfgs_eigh.cpp", "bfgs_eigh.cu"):
            code = (source / name).read_text()
            if args.method == "jacobi" and name.endswith(".cu"):
                code = code.replace("if (n < 200)", "if (true)")
            (diagnostic / name).write_text(code)
        extension = load_hip_extension(
            source_root=diagnostic,
            extension_name=f"nvalchemi_bfgs_{args.method}_diagnostic",
            source_names=("bfgs_eigh.cpp", "bfgs_eigh.cu"),
            build_env_var="NVALCHEMI_BFGS_JACOBI_DIAGNOSTIC_BUILD_DIR",
            verbose_env_var="NVALCHEMI_BFGS_JACOBI_DIAGNOSTIC_VERBOSE",
            extra_ldflags=("-lrocsolver", "-lrocblas"),
        )

        def solve(a):
            w, v, info, residual, sweeps = extension.eigh(a, "L", 100)
            assert not bool(info.any()), info.tolist()
            return w, v
    elif args.method == "torch":
        solve = torch.linalg.eigh
    else:
        solve = _eigh
    torch.cuda.synchronize()
    start = time.perf_counter()
    w, v = solve(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    residual = ((v * w.unsqueeze(-2)) @ v.mT - a).norm() / a.norm()
    orthogonal = (
        v.mT @ v - torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    ).norm()
    torch.testing.assert_close(w.cpu(), expected, atol=2e-10, rtol=2e-11)
    assert residual.item() < 1e-11 and orthogonal.item() < 1e-11
    summary = dict(
        status="passed",
        method=args.method,
        shape=list(a.shape),
        matrix_sha256=hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        eigenvalue_error=(w.cpu() - expected).abs().max().item(),
        relative_residual=residual.item(),
        orthogonality_error=orthogonal.item(),
        elapsed_s=elapsed,
        near_alpha_eigenvalues=int(((expected - 70).abs() < 1e-8).sum()),
        dtype=str(a.dtype),
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
