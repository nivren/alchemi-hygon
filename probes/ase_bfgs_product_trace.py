# SPDX-License-Identifier: Apache-2.0
"""Bounded product HIP timeline and sanitized overlap/transfer analysis."""

import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
prefix = a.root / "trace-product"
metadata = a.root / "trace-product-metadata.json"
with (a.root / "trace-product.log").open("w") as log:
    result = subprocess.run(
        [
            "/opt/dtk-26.04/bin/hipprof",
            "--hip-trace",
            "--trace-args",
            "-o",
            str(prefix),
            sys.executable,
            "probes/ase_bfgs_product_benchmark.py",
            "--real-oracles",
            "artifacts/ase-bfgs-m5-3a-2026-10-10/real-hessian-oracles.pt",
            "--trace",
            "product",
            "--input-set",
            "li",
            "--output",
            str(metadata),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
if result.returncode:
    raise SystemExit(result.returncode)
result = subprocess.run(
    [
        sys.executable,
        "probes/ase_bfgs_heterogeneous_trace_summary.py",
        "--database",
        str(prefix.with_suffix(".db")),
        "--metadata",
        str(metadata.with_suffix(".result.json")),
        "--output",
        str(a.root / "trace-product-summary.json"),
    ],
    timeout=30,
)
raise SystemExit(result.returncode)
