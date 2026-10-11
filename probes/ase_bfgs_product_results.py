# SPDX-License-Identifier: Apache-2.0
"""Aggregate retained M5-4 samples without replacing any raw acceptance inputs."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--benchmark-prefix", default="")
p.add_argument("--output", type=Path)
a = p.parse_args()
root = a.root
benchmarks = [
    json.loads((root / f"{a.benchmark_prefix}benchmark-{i}.json").read_text())
    for i in (1, 2, 3)
]
assert all(r["status"] == "passed" for r in benchmarks)
result = {
    "status": "passed",
    "benchmark_prefix": a.benchmark_prefix,
    "benchmarks": [],
    "csp": {},
    "csp_revision": "pre-zero-sweep-fix; iterative kernels unchanged",
}
for scope in [c["scope"] for c in benchmarks[0]["cases"]]:
    rows = [next(c for c in r["cases"] if c["scope"] == scope) for r in benchmarks]
    summary = {"scope": scope, "shapes": rows[0]["shapes"], "windows": {}}
    for window in ("checked_solver", "complete_math"):
        values = {}
        for variant in ("rocsolver", "serial", "product"):
            samples = [
                v for row in rows for v in row[window]["wall_ms"][variant]["samples"]
            ]
            assert len(samples) == 21
            values[variant] = {
                "samples": 21,
                "median_ms": statistics.median(samples),
                "minimum_ms": min(samples),
                "maximum_ms": max(samples),
            }
        summary["windows"][window] = values
    result["benchmarks"].append(summary)
old = [json.loads((root / f"csp-rocsolver-{i}.json").read_text()) for i in (1, 2, 3)]
new = [
    json.loads((root / f"{tag}.json").read_text())
    for tag in ("csp-product-warm-1", "csp-product-2", "csp-product-3")
]
assert all(r["all_converged"] for r in old + new)
assert all(r["stage_steps"] == 233 and r["model_evaluations"] == 234 for r in old + new)
for name, runs in (("rocsolver", old), ("product", new)):
    result["csp"][name] = {
        "elapsed_s": [r["elapsed_s"] for r in runs],
        "elapsed_median_s": statistics.median(r["elapsed_s"] for r in runs),
        "update_total_s": [r["timing_s"]["bfgs_update"]["total"] for r in runs],
        "update_total_median_s": statistics.median(
            r["timing_s"]["bfgs_update"]["total"] for r in runs
        ),
        "scope": "Acceptance probe includes oracle checks and cold initialization, uses cached native compilation; not production throughput",
    }
result["csp"]["first_product_with_cold_build_s"] = json.loads(
    (root / "csp-product-1.json").read_text()
)["elapsed_s"]
output = a.output or root / "results-summary.json"
if output.exists():
    raise FileExistsError(f"Refusing to replace retained summary: {output}")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
