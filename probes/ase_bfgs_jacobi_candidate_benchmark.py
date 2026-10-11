# SPDX-License-Identifier: Apache-2.0
"""Alternating, equal-scope comparison of the isolated solver and rocSOLVER."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ase_bfgs_jacobi_candidate import checked_raw, eigh as candidate, load
from ase_bfgs_jacobi_candidate_test import validate
from nvalchemiops._bfgs_eigh_hip import _eigh as baseline, _load_extension
from nvalchemiops.bfgs_ase import bfgs_ase_prepare_state
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh
from nvalchemiops.bfgs_ase_reference import _check_eigensystem


def stats(values):
    return dict(samples=values, median=statistics.median(values), minimum=min(values),
                maximum=max(values), mean=statistics.mean(values), stdev=statistics.pstdev(values))


def paired(left, right, repeats, reset=lambda _:None):
    functions={"rocsolver":left,"candidate":right}
    cold={}
    for name,fn in functions.items():
        reset(name)
        torch.cuda.synchronize()
        start=time.perf_counter()
        fn()
        torch.cuda.synchronize()
        cold[name]=(time.perf_counter()-start)*1000
    for _ in range(2):
        for name,fn in functions.items():
            reset(name)
            fn()
    torch.cuda.synchronize()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    wall={name:[] for name in functions}
    stream={name:[] for name in functions}
    for repeat in range(repeats):
        order=list(functions) if repeat%2==0 else list(reversed(functions))
        for name in order:
            reset(name)
            torch.cuda.synchronize()
            start=time.perf_counter()
            begin.record()
            functions[name]()
            end.record()
            end.synchronize()
            wall[name].append((time.perf_counter()-start)*1000)
            stream[name].append(begin.elapsed_time(end))
    return dict(first_call_ms=cold,wall_ms={k:stats(v) for k,v in wall.items()},
                stream_elapsed_ms={k:stats(v) for k,v in stream.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--trace",action="store_true")
    parser.add_argument("--n",type=int,choices=(147,285),default=285)
    parser.add_argument("--batch",type=int,choices=(1,4),default=1)
    args=parser.parse_args()
    if args.repeats<7:
        parser.error("at least seven samples required")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    cases=torch.load(args.dataset,map_location="cpu",weights_only=True)
    start=time.perf_counter()
    load()
    module=_load_extension()
    result=dict(status="running",extension_load_s=time.perf_counter()-start,
                torch_hcu=getattr(torch.version,"__hcu_version__",None),hip=torch.version.hip,
                rocsolver=module.rocsolver_version(),device=torch.cuda.get_device_name(),
                warmup=2,repeats=args.repeats,cases=[])
    if args.trace:
        captured=next(c for c in cases if c["n"]==args.n and c["name"]=="captured")
        a=captured["input"].repeat(args.batch,1,1).cuda()
        for _ in range(2):
            checked_raw(a)
        torch.cuda.synchronize()
        begin=time.time_ns()
        outputs=checked_raw(a)
        torch.cuda.synchronize()
        end=time.time_ns()
        result.update(status="passed",n=args.n,batch=args.batch,window_unix_ns=[begin,end],
                      scope="candidate checked solver; oracle reads outside window",
                      correctness=validate(a,outputs,[captured]*args.batch))
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(dict(status="passed",mode="trace")),flush=True)
        return
    for n in (147,285):
        captured=next(c for c in cases if c["n"]==n and c["name"]=="captured")
        for batch,scope in [(1,"captured"),(4,"identical_captured"),(4,"distinct")]:
            references=[captured]*batch if scope!="distinct" else [
                next(c for c in cases if c["n"]==n and c["name"]==name)
                for name in ("captured","random","clustered","diagonal")]
            a=torch.stack([c["input"] for c in references]).cuda()
            original=a.clone()
            row=dict(n=n,batch=batch,scope=scope)
            for name,fn in [("rocsolver",baseline),("candidate",candidate)]:
                w,v=fn(a)
                row[name+"_correctness"]=validate(a,(w,v),references)
            row["checked_solver"]=paired(lambda:baseline(a),lambda:candidate(a),args.repeats)
            def guarded(fn):
                w,v=fn(a)
                _check_eigensystem(a,w,v)
            row["solver_plus_validity"]=paired(lambda:guarded(baseline),lambda:guarded(candidate),args.repeats)
            assert torch.equal(original,a)
            result["cases"].append(row)
            print(json.dumps(dict(n=n,batch=batch,scope=scope,wall_ms={
                k:v["median"] for k,v in row["checked_solver"]["wall_ms"].items()})),flush=True)
    # Equal-scope direct grouped math; excludes public dispatcher validation.
    widths=[147,285,147]
    pools={}
    for name in ("rocsolver","candidate"):
        initial=[torch.linspace(-.1,.2,n,device="cuda",dtype=torch.float64).reshape(-1,3) for n in widths]
        qs=[q.clone() for q in initial]
        forces=[-q.clone() for q in initial]
        states=[bfgs_ase_prepare_state(n,device="cuda",system_id=i,maxstep=.2) for i,n in enumerate(widths)]
        h=[next(c["input"] for c in cases if c["n"]==n and c["name"]=="captured").cuda() for n in widths]
        pools[name]=(initial,qs,forces,states,h)
    def reset(name):
        initial,qs,forces,states,hs=pools[name]
        for start,q,s,h in zip(initial,qs,states,hs):
            q.copy_(start)
            s.hessian.copy_(h)
            s.previous_positions.zero_()
            s.previous_forces.zero_()
            s.initialized,s.step_count=False,0
    def group(name,fn):
        _,qs,forces,states,_=pools[name]
        return _step_batch_with_eigh(qs,forces,states,fn)
    result["group"]=paired(lambda:group("rocsolver",baseline),lambda:group("candidate",candidate),args.repeats,reset)
    for actual,reference in zip(pools["candidate"][1],pools["rocsolver"][1]):
        torch.testing.assert_close(actual,reference,atol=1e-10,rtol=1e-8)
    result.update(status="passed",group_widths=widths,group_scope="direct math; captured H plus synthetic q/f; history reset")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status="passed",group_wall_ms={k:v["median"] for k,v in result["group"]["wall_ms"].items()})),flush=True)


if __name__=="__main__":
    main()
