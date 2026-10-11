# SPDX-License-Identifier: Apache-2.0
"""Equal-instrumentation CSP comparison; solver patch exists in this process only."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ase_bfgs_csp_profile
from nvalchemiops._bfgs_eigh_hip import _load_extension


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--solver",choices=("rocsolver","candidate"),required=True)
    options,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    module=_load_extension()
    original=module.eigh
    if options.solver=="candidate":
        from ase_bfgs_jacobi_candidate import load, raw_eigh
        load()
        def candidate_native(a,uplo,max_sweeps):
            if uplo!="L":
                raise ValueError("isolated candidate supports only LOWER")
            return raw_eigh(a,max_sweeps=max_sweeps)
        module.eigh=candidate_native
    try:
        ase_bfgs_csp_profile.main()
    finally:
        module.eigh=original
    output=Path(sys.argv[sys.argv.index("--output")+1])
    for path in [output,output.with_suffix(".components.json")]:
        data=json.loads(path.read_text())
        data["actual_solver"]=options.solver
        data["solver_injection_scope"]="unregistered process-only native function patch"
        data["candidate_native_sha256"]=hashlib.sha256(
            Path(__file__).with_name("native").joinpath("ase_jacobi_candidate.cu").read_bytes()).hexdigest()
        path.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(dict(status="passed",actual_solver=options.solver)),flush=True)


if __name__=="__main__":
    main()
