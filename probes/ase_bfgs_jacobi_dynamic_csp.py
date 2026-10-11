# SPDX-License-Identifier: Apache-2.0
"""M5-3A process-only solver injection for user-provided crystal qualification."""

import argparse
import json
import sys
from pathlib import Path

import torch

import ase_bfgs_csp
from nvalchemiops._bfgs_eigh_hip import _load_extension


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-set", choices=("li", "t2"), required=True)
    parser.add_argument("--solver", choices=("rocsolver", "candidate"), default="candidate")
    parser.add_argument("--capture-groups", type=Path)
    args, remaining = parser.parse_known_args()
    inputs = json.loads(args.manifest.read_text())[args.input_set]
    sys.argv = [sys.argv[0], *remaining, "--cifs", *[row["path"] for row in inputs]]
    extension = _load_extension()
    original = extension.eigh
    groups = []
    if args.solver == "candidate":
        from ase_bfgs_jacobi_candidate import load, raw_eigh
        load()
    if args.capture_groups is not None:
        args.capture_groups.mkdir(parents=True, exist_ok=True)
    def injected(a, uplo, max_sweeps):
        groups.append(list(a.shape))
        if args.capture_groups is not None:
            torch.save(a.cpu(), args.capture_groups / f"last-{a.shape[-1]}.pt")
        if args.solver == "candidate":
            if uplo != "L":
                raise ValueError("candidate supports only LOWER")
            return raw_eigh(a,max_sweeps=max_sweeps)
        return original(a,uplo,max_sweeps)
    extension.eigh = injected
    try:
        ase_bfgs_csp.main()
    finally:
        extension.eigh = original
    output = Path(sys.argv[sys.argv.index("--output")+1])
    result = json.loads(output.read_text())
    result.update(actual_solver=args.solver, solver_injection_scope="unregistered process-only native function patch",
                  observed_group_shapes=groups, input_set=args.input_set,
                  diagnostic_matrix_capture=args.capture_groups is not None,
                  performance_acceptance=False, manifest=str(args.manifest))
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status=result["status"],input_set=args.input_set,groups=groups)),flush=True)


if __name__ == "__main__":
    main()
