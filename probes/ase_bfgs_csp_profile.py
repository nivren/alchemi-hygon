# SPDX-License-Identifier: Apache-2.0
"""M5-1 timing instrumentation around the unchanged M4 CSP acceptance workflow.

All monkey patches are confined to this diagnostic process. Raw native wall time
ends at CPU return; checked eigh and validity guards include their existing scalar
synchronization. Nested durations are not summed as independent total costs.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ase_bfgs_csp
import nvalchemiops.bfgs_ase_batch_reference as batch_math
import nvalchemiops.bfgs_ase_hip as hip_math
import nvalchemiops.bfgs_ase_reference as single_math
from nvalchemiops._bfgs_eigh_hip import _load_extension


def main():
    if "--device" not in sys.argv or sys.argv[sys.argv.index("--device") + 1] != "cuda":
        raise ValueError("M5-1 device timing requires --device cuda")
    if "--backend" not in sys.argv or sys.argv[sys.argv.index("--backend") + 1] != "hip":
        raise ValueError("M5-1 solver timing requires explicit --backend hip")
    if "--capture-eigh" in sys.argv:
        raise ValueError("CPU matrix capture is excluded from performance measurements")
    path = Path(sys.argv[sys.argv.index("--output") + 1])
    samples = {"native_cpu_return": [], "checked_eigh": [], "eigensystem_guard": []}
    metadata = []
    extension = _load_extension()
    native, checked, guard = extension.eigh, hip_math._eigh, single_math._check_eigensystem

    def timed_native(a, uplo, max_sweeps):
        begin = time.perf_counter()
        outputs = native(a, uplo, max_sweeps)
        samples["native_cpu_return"].append(dict(n=a.shape[-1], batch=1 if a.ndim == 2 else a.shape[0],
                                                seconds=time.perf_counter() - begin))
        metadata.append((a.shape[-1], outputs[2], outputs[3], outputs[4]))
        return outputs

    def timed_checked(a, **kwargs):
        begin = time.perf_counter()
        outputs = checked(a, **kwargs)
        samples["checked_eigh"].append(dict(n=a.shape[-1], batch=1 if a.ndim == 2 else a.shape[0],
                                           seconds=time.perf_counter() - begin))
        return outputs

    def timed_guard(a, w, v):
        begin = time.perf_counter()
        guard(a, w, v)
        samples["eigensystem_guard"].append(dict(n=a.shape[-1], batch=1 if a.ndim == 2 else a.shape[0],
                                               seconds=time.perf_counter() - begin))

    extension.eigh = timed_native
    hip_math._eigh = timed_checked
    single_math._check_eigensystem = batch_math._check_eigensystem = timed_guard
    try:
        ase_bfgs_csp.main()
    finally:
        extension.eigh, hip_math._eigh = native, checked
        single_math._check_eigensystem = batch_math._check_eigensystem = guard
    torch.cuda.synchronize()
    summary = {}
    for phase, rows in samples.items():
        groups = {}
        for row in rows:
            groups.setdefault(f"n{row['n']}-b{row['batch']}", []).append(row["seconds"])
        summary[phase] = {group: dict(count=len(values), total_s=sum(values),
                                     median_ms=statistics.median(values)*1000)
                          for group, values in groups.items()}
    library_diagnostics = [dict(n=n, info=info.tolist(), residual=residual.tolist(), sweeps=sweeps.tolist())
                           for n, info, residual, sweeps in metadata]
    assert all(all(x == 0 for x in row["info"]) for row in library_diagnostics)
    result = dict(status="passed", scope="instrumented unchanged M4 CSP; nested solver timings",
                  torch_hcu=getattr(torch.version, "__hcu_version__", None),
                  hip=torch.version.hip, rocsolver=extension.rocsolver_version(),
                  summary=summary, samples=samples, library_diagnostics=library_diagnostics,
                  extension_loaded_before_workflow=True,
                  normal_end_to_end_timing=False)
    path.with_suffix(".components.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"components_status": "passed", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
