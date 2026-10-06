# SPDX-License-Identifier: Apache-2.0
"""M6 native-query isolation with the existing M5 three-path fixtures."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch

import hip_batch_metadata_benchmark as synthetic
import nvalchemiops._hip_batch_cell_query as query
from hip_neighbor_workspace_benchmark import stage_times
from nvalchemiops._hip_extension_loader import load_hip_extension


ROOT = Path("artifacts/hip-cooperative-query-m6")


class QueryPaths(synthetic.ThreePaths):
    """Select archived M5 query or current query outside API timing."""

    def __init__(self, baseline):
        super().__init__(baseline)
        self.candidate_query = query._load_jit_extension()
        self.baseline_query = load_hip_extension(
            source_root=ROOT / "baseline-native",
            extension_name="nvalchemi_m6_serial_query",
            source_names=("batch_cell_query.cpp", "batch_cell_query.cu"),
            build_env_var="NVALCHEMI_M6_SERIAL_QUERY_BUILD_DIR",
            verbose_env_var="NVALCHEMI_M6_SERIAL_QUERY_VERBOSE",
        )
        assert self.baseline_query is not self.candidate_query
        assert self.baseline_query.__file__ != self.candidate_query.__file__
        print("independent native modules", self.baseline_query.__file__, self.candidate_query.__file__, flush=True)

    def select(self, path):
        query._EXTENSION = self.baseline_query if path == "hip_original" else self.candidate_query
        return super().select(path)


def main():
    """Replay synthetic/real fixtures or collect separately synchronized stages."""
    if "--stages-only" in sys.argv:
        ROOT.mkdir(parents=True, exist_ok=True)
        paths = QueryPaths(synthetic.load_baseline(ROOT / "baseline-executor.py"))
        import nvalchemi.neighbors as framework_neighbors
        report = []
        for dtype in (torch.float32, torch.float64):
            for case, size, count in (("many", 100, 200), ("medium", 500, 40), ("few", 2000, 10), ("small", 46, 32)):
                _, batch = next(synthetic.snapshots(size, count, dtype, seed=20261007 if case == "small" else 20261006))
                paths.select("hip_optimized")
                if "--stages-original" in sys.argv:
                    query._EXTENSION = paths.baseline_query
                call = lambda: paths.neighbor(batch, "hip_optimized", 6.0)
                call()
                stages = []
                for _ in range(3):
                    coo_ms = []
                    original = framework_neighbors._neighbor_matrix_to_coo

                    def coo(*args, **kwargs):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = original(*args, **kwargs)
                        torch.cuda.synchronize()
                        coo_ms.append((time.perf_counter() - start) * 1000)
                        return result

                    with patch.object(framework_neighbors, "_neighbor_matrix_to_coo", coo):
                        measured = stage_times(call)
                    measured["framework_matrix_to_coo"] = coo_ms
                    stages.append(measured)
                report.append({"case": case, "dtype": str(dtype), "stages_ms": stages})
                print(case, dtype, stages[0], flush=True)
        (ROOT / (os.environ.get("M6_STAGE_LABEL", "baseline") + "-stages.json")).write_text(json.dumps(report, indent=2) + "\n")
        return
    synthetic.ThreePaths = QueryPaths
    if "--real" in sys.argv:
        sys.argv.remove("--real")
        import hip_batch_metadata_mace as real
        real.ThreePaths = QueryPaths
        real.main()
    else:
        synthetic.main()


if __name__ == "__main__":
    main()
