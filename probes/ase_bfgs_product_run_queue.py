# SPDX-License-Identifier: Apache-2.0
"""Run an explicit sequential single-card acceptance queue with per-job bounds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--queue", type=Path, required=True)
a = p.parse_args()
root = a.queue.parent
for name, bound, arguments in json.loads(a.queue.read_text()):
    command = [
        sys.executable,
        "probes/ase_bfgs_heterogeneous_run.py",
        "--record",
        str(root / (name + "-run.json")),
        "--timeout",
        str(bound),
        "--",
        sys.executable,
        *arguments,
    ]
    with (root / (name + ".log")).open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=bound + 30
        )
    print(json.dumps(dict(job=name, exit_code=result.returncode)), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
