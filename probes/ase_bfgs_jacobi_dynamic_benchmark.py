# SPDX-License-Identifier: Apache-2.0
"""M5-3A frozen-prototype parity, dynamic paired timings, and device trace."""

import argparse
import json
import time
from pathlib import Path

import torch

from ase_bfgs_jacobi_candidate import checked_raw, eigh, load
from ase_bfgs_jacobi_candidate_benchmark import paired
from ase_bfgs_jacobi_candidate_test import validate
from nvalchemiops._bfgs_eigh_hip import _eigh as baseline, _load_extension
from nvalchemiops._hip_extension_loader import load_hip_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("frozen", "benchmark", "trace"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--real-oracles", type=Path,
                        help="CPU oracle rows, each containing a whole captured real group")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error("at least seven checked timing samples required")
    torch.set_num_threads(1)
    cases = torch.load(args.dataset, map_location="cpu", weights_only=True)
    load()
    result = dict(status="running", mode=args.mode, torch=str(torch.__version__),
                  hip=torch.version.hip, device=torch.cuda.get_device_name(), cases=[])
    if args.mode == "frozen":
        frozen = load_hip_extension(
            source_root=args.frozen_source, extension_name="ase_bfgs_jacobi_m52_frozen",
            source_names=("ase_jacobi_candidate.cpp", "ase_jacobi_candidate.cu"),
            build_env_var="ASE_JACOBI_FROZEN_BUILD_DIR", verbose_env_var="ASE_JACOBI_PROBE_BUILD_VERBOSE")
        for n in (147,285):
            for batch in (1,4):
                refs = [next(c for c in cases if c["n"] == n and c["name"] == name)
                        for name in ("captured", "random", "clustered", "diagonal")[:batch]]
                a = torch.stack([c["input"] for c in refs]).cuda()
                old, new = frozen.eigh(a,100), checked_raw(a)
                assert all(torch.equal(x,y) for x,y in zip(old,new))
                result["cases"].append(dict(n=n,batch=batch,all_five_outputs_bitwise_equal=True))
    elif args.mode == "trace":
        ref = next(c for c in cases if c["n"] == 513)
        a = ref["input"].unsqueeze(0).cuda()
        for _ in range(2):
            checked_raw(a)
        torch.cuda.synchronize()
        begin = time.time_ns()
        outputs = checked_raw(a)
        torch.cuda.synchronize()
        end = time.time_ns()
        result.update(n=513,batch=1,window_unix_ns=[begin,end],
                      scope="checked candidate; matrix uploads and numerical oracle reads outside window",
                      correctness=validate(a,outputs,[ref]))
    else:
        module = _load_extension()
        result.update(rocsolver=module.rocsolver_version(),warmup=2,repeats=args.repeats)
        for n,batch in ((189,16),(199,32),(513,1),(513,8),(561,1),(561,8)):
            ref = next(c for c in cases if c["n"] == n)
            a = ref["input"].repeat(batch,1,1).cuda()
            row = dict(n=n,batch=batch,scope="repeated synthetic dense matrix; full checked solver")
            row["baseline_correctness"] = validate(a,baseline(a),[ref]*batch)
            row["candidate_correctness"] = validate(a,checked_raw(a),[ref]*batch)
            row["torch_allocator_peak_extra_bytes"] = {}
            for name, fn in (("rocsolver",baseline),("candidate",eigh)):
                start_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                outputs = fn(a)
                torch.cuda.synchronize()
                row["torch_allocator_peak_extra_bytes"][name] = torch.cuda.max_memory_allocated()-start_bytes
                del outputs
            row["paired"] = paired(lambda:baseline(a),lambda:eigh(a),args.repeats)
            if n == 513 and batch == 8:
                row["chunk3"] = paired(lambda:eigh(a),lambda:eigh(a,chunk_size=3),args.repeats)
                row["chunk3_labels"] = dict(rocsolver="candidate full group",candidate="candidate chunk_size=3")
            result["cases"].append(row)
            print(json.dumps(dict(n=n,batch=batch,wall_ms={k:v["median"] for k,v in row["paired"]["wall_ms"].items()})),flush=True)
        if args.real_oracles is not None:
            real = torch.load(args.real_oracles,map_location="cpu",weights_only=True)
            for scope in ("li", "t2"):
                references = [row for row in real if row["scope"] == scope]
                groups = [row["input"].cuda() for row in references]
                row = dict(scope=scope+" captured real Hessians; same-stream serial groups",
                           shapes=[list(a.shape) for a in groups],group_correctness=[])
                for a,ref in zip(groups,references):
                    row["group_correctness"].append(validate(a,checked_raw(a),ref["references"]))
                    validate(a,baseline(a),ref["references"])
                def serial(fn):
                    return [fn(a) for a in groups]
                row["paired"] = paired(lambda:serial(baseline),lambda:serial(eigh),args.repeats)
                result["cases"].append(row)
                print(json.dumps(dict(scope=scope,wall_ms={k:v["median"] for k,v in row["paired"]["wall_ms"].items()})),flush=True)
    result["status"] = "passed"
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    if args.mode == "trace":
        # Keep metadata independently of a profiler exporter sharing the prefix.
        args.output.with_suffix(".result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status="passed",mode=args.mode,cases=len(result["cases"]))),flush=True)


if __name__ == "__main__":
    main()
