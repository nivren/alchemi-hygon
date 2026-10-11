# SPDX-License-Identifier: Apache-2.0
"""Independent CPU oracle preparation and isolated HIP Jacobi qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def pairs(n, round_index):
    m = n + n % 2
    vertices = [0] + [1 + (i + round_index) % (m - 1) for i in range(m - 1)]
    return [(p, q) for p, q in zip(vertices[:m//2], reversed(vertices[m//2:]))
            if p < n and q < n]


def cpu_reference_gate():
    """All unordered pairs once/sweep, and an independent dense similarity test."""
    for n in (7, 147, 285):
        visits = []
        for r in range(n + n % 2 - 1):
            matching = pairs(n, r)
            endpoints = [i for pair in matching for i in pair]
            assert len(endpoints) == len(set(endpoints))
            visits.extend(tuple(sorted(pair)) for pair in matching)
        assert len(visits) == len(set(visits)) == n*(n-1)//2
    torch.manual_seed(517)
    h = torch.randn(7, 7, dtype=torch.float64)
    h = (h + h.T)/2
    a, v = h.clone(), torch.eye(7, dtype=torch.float64)
    for _ in range(12):
        for r in range(7):
            rotation = torch.eye(7, dtype=torch.float64)
            for p, q in pairs(7, r):
                b = a[p, q]
                if b == 0:
                    continue
                delta = (a[q, q] - a[p, p])/2
                t = b/(delta + torch.copysign(torch.hypot(delta, b), delta))
                c = torch.rsqrt(1 + t*t)
                s = t*c
                rotation[p, p] = rotation[q, q] = c
                rotation[p, q], rotation[q, p] = s, -s
            a = rotation.T @ a @ rotation
            v = v @ rotation
    w, permutation = a.diag().sort()
    v = v[:, permutation]
    torch.testing.assert_close(w, torch.linalg.eigvalsh(h), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(h@v, v*w, atol=1e-12, rtol=1e-12)
    return dict(status="passed", dimensions=[7, 147, 285], numerical_dimension=7)


def prepare(args):
    gate = cpu_reference_gate()
    cases, hashes = [], {}
    for n in (147, 285):
        path = args.source/f"last-{n}.pt"
        captured = torch.load(path, map_location="cpu", weights_only=True).reshape(n, n)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        generator = torch.Generator().manual_seed(1810 + n)
        r = torch.randn(n, n, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(r)
        cluster = torch.full((n,), 70.0, dtype=torch.float64)
        cluster[:8] = torch.tensor([-40, -1, 0.5, 1, 3, 10, 35, 80], dtype=torch.float64)
        near = 70 + torch.linspace(-1e-8, 1e-8, n, dtype=torch.float64)
        values = {"captured": captured, "random": (r+r.T)/2,
                  "known_spectrum": (q*torch.linspace(-80, 80, n, dtype=torch.float64))@q.T,
                  "clustered": (q*cluster)@q.T, "near_degenerate": (q*near)@q.T,
                  "diagonal": torch.diag(torch.linspace(-80, 80, n, dtype=torch.float64)),
                  "zero": torch.zeros(n, n, dtype=torch.float64)}
        values["upper_poison"] = torch.where(torch.triu(torch.ones(n, n, dtype=torch.bool), 1),
                                             torch.full_like(captured, float("nan")), captured)
        for name, a in values.items():
            symmetric = a.tril() + a.tril(-1).T
            w, v = torch.linalg.eigh(symmetric)
            cases.append(dict(name=name, n=n, input=a, symmetric=symmetric, w=w, v=v))
    if args.regression_matrix is not None:
        path=args.regression_matrix
        a=torch.load(path,map_location="cpu",weights_only=True).reshape(285,285)
        symmetric=a.tril()+a.tril(-1).T
        w,v=torch.linalg.eigh(symmetric)
        cases.append(dict(name="bfgs_step2_regression",n=285,input=a,symmetric=symmetric,w=w,v=v))
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(dict(cpu_gate=gate, inputs=hashes,
                                                  cases=len(cases)), indent=2)+"\n")
    print(json.dumps(dict(status="passed", cpu_gate=gate, cases=len(cases))), flush=True)


def validate(a, outputs, references):
    w,v=outputs[:2]
    diagnostics={}
    if len(outputs)==5:
        _,_,info,residual,sweeps=outputs
        assert bool((info == 0).all()), (info.tolist(), residual.tolist(), sweeps.tolist())
        diagnostics=dict(info=info.tolist(),residual=residual.tolist(),sweeps=sweeps.tolist())
    if a.ndim == 2:
        a, w, v = a.unsqueeze(0), w.unsqueeze(0), v.unsqueeze(0)
    target = torch.stack([case["w"] for case in references])
    h = torch.stack([case["symmetric"] for case in references]).cuda()
    torch.testing.assert_close(w.cpu(), target, atol=2e-10, rtol=2e-11)
    equation = ((h@v-v*w.unsqueeze(-2)).norm(dim=(-2, -1))/h.norm(dim=(-2, -1)).clamp_min(1)).max().item()
    orthogonal = ((v.mT@v-torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)).norm(dim=(-2, -1))
                  / a.shape[-1]**0.5).max().item()
    assert equation < 1e-10 and orthogonal < 1e-10, (equation, orthogonal)
    return dict(spectrum_abs=(w.cpu()-target).abs().max().item(), equation_relative=equation,
                orthogonal_relative=orthogonal, **diagnostics)


def correctness(args):
    from ase_bfgs_jacobi_candidate import checked_raw, eigh, load, raw_eigh
    from nvalchemiops.bfgs_ase import BFGSNumericalError

    cases = torch.load(args.dataset, map_location="cpu", weights_only=True)
    torch.cuda.set_device(0)
    start = time.perf_counter()
    load()
    result = dict(status="running", extension_load_s=time.perf_counter()-start,
                  torch_hcu=getattr(torch.version, "__hcu_version__", None), hip=torch.version.hip,
                  device=torch.cuda.get_device_name(), cases=[])
    for case in cases:
        a = case["input"].cuda()
        before = a.view(torch.int64).clone()
        outputs = checked_raw(a)
        error = validate(a, outputs, [case])
        assert torch.equal(before, a.view(torch.int64))
        assert outputs[0].is_contiguous() and outputs[1].is_contiguous()
        assert all(out.device == a.device for out in outputs)
        assert outputs[1].data_ptr() != a.data_ptr()
        result["cases"].append(dict(n=case["n"], name=case["name"], batch=1, **error))
        print(json.dumps(result["cases"][-1]), flush=True)
    for n in (147, 285):
        selected = [next(case for case in cases if case["n"] == n and case["name"] == name)
                    for name in ("captured", "random", "clustered", "zero")]
        a = torch.stack([case["input"] for case in selected]).cuda()
        before = a.clone()
        result["cases"].append(dict(n=n, name="distinct_batch_and_freeze", batch=4,
                                    **validate(a, checked_raw(a), selected)))
        assert torch.equal(before, a)
    reference = next(c for c in cases if c["n"] == 147 and c["name"] == "random")
    a = reference["input"].cuda()
    outputs = raw_eigh(a, max_sweeps=1)
    assert outputs[2].tolist() == [1] and outputs[4].tolist() == [1]
    try:
        checked_raw(a, max_sweeps=1)
    except BFGSNumericalError:
        pass
    else:
        raise AssertionError("real nonconvergence not rejected")
    rejected = 0
    invalid = [(a.cpu(), 30), (a.float(), 30), (a.T, 30), (a[:,:146].contiguous(), 30),
               (a[:0,:0], 30), (a.unsqueeze(0)[:0], 30),
               (a.clone().requires_grad_(), 30), (a, 0), (a, True), (a, 101), (a, 1.5)]
    for value, sweeps in invalid:
        try:
            raw_eigh(value, sweeps)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid input admitted")
    try:
        eigh(a, UPLO="U")
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("UPPER unexpectedly admitted")
    poisoned = a.clone()
    poisoned[1, 0] = float("nan")
    try:
        checked_raw(poisoned)
    except BFGSNumericalError:
        rejected += 1
    else:
        raise AssertionError("selected triangle nonfinite admitted")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stream_input = a.clone()
        outputs = checked_raw(stream_input)
        event = torch.cuda.Event()
        event.record()
    torch.cuda.current_stream().wait_event(event)
    result["stream"] = validate(stream_input, outputs, [reference])
    result.update(status="passed", rejected_inputs=rejected, actual_nonconvergence=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status="passed", cases=len(result["cases"]), rejected_inputs=rejected)), flush=True)


def bfgs_gate(args):
    import io
    import numpy as np
    from ase import Atoms
    from ase.optimize import BFGS
    from ase_bfgs_jacobi_candidate import checked_raw, eigh, load, raw_eigh
    from nvalchemiops.bfgs_ase import ASEBFGSState, BFGSNumericalError, bfgs_ase_prepare_state
    from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh
    from nvalchemiops.bfgs_ase_reference import _step_with_eigh

    load()
    widths = args.widths
    if len(widths) < 2 or any(n < 3 or n % 3 for n in widths):
        raise ValueError("BFGS gate requires at least two positive Cartesian widths divisible by three")
    qs = [torch.linspace(-0.4, 0.6+i*0.01, n, device="cuda", dtype=torch.float64).reshape(-1,3)
          for i,n in enumerate(widths)]
    states = [bfgs_ase_prepare_state(n, device="cuda", system_id=i, alpha=53, maxstep=0.12)
              for i,n in enumerate(widths)]
    opts = [BFGS(Atoms("H"*(n//3), positions=q.cpu().numpy()), logfile=None, alpha=53, maxstep=0.12)
            for n,q in zip(widths,qs)]

    def force(q):
        flat = q.flatten()
        k = torch.linspace(4,14,flat.numel(), device=q.device, dtype=q.dtype)
        return -(k*flat+0.7*flat.sum()+0.6*flat**3).reshape_as(q)

    errors = {k:0.0 for k in ["coordinates","hessian","spectrum","raw_step","applied_step"]}
    shapes = []
    restart_qs, restart_states = None, None
    def recorded_eigh(a, **kwargs):
        shapes.append(list(a.shape))
        try:
            return eigh(a, **kwargs)
        except BFGSNumericalError:
            torch.save(a.cpu(),args.output.with_suffix(".failed-matrix.pt"))
            outputs=raw_eigh(a,max_sweeps=100)
            print(json.dumps(dict(failed_group=list(a.shape),step=step,
                                  limit100_info=outputs[2].tolist(),residual=outputs[3].tolist(),
                                  sweeps=outputs[4].tolist())),flush=True)
            raise
    for step in range(10):
        results = _step_batch_with_eigh(qs, [force(q) for q in qs], states, recorded_eigh)
        for q,state,opt,output in zip(qs,states,opts,results):
            old = opt.atoms.positions.copy()
            x = old.ravel()
            f = -(np.linspace(4,14,x.size)*x+0.7*x.sum()+0.6*x**3)
            raw, lengths = opt.prepare_step(x, -f)
            applied = opt.determine_step(raw.copy(), lengths)
            opt.atoms.set_positions(old + applied.reshape(-1,3))
            targets = [("coordinates",q,opt.atoms.positions,3e-12,3e-12),
                       ("hessian",state.hessian,opt.H,3e-10,3e-11),
                       ("spectrum",output.eigenvalues,np.linalg.eigvalsh(opt.H),2e-10,2e-11),
                       ("raw_step",output.raw_step,raw.reshape(-1,3),1e-12,1e-10),
                       ("applied_step",output.applied_step,applied.reshape(-1,3),1e-12,1e-10)]
            for key,value,target,atol,rtol in targets:
                actual=value.cpu().numpy()
                np.testing.assert_allclose(actual,target,atol=atol,rtol=rtol,err_msg=f"{key} step{step}")
                errors[key]=max(errors[key],float(np.abs(actual-target).max()))
        if restart_qs is not None:
            _step_batch_with_eigh(restart_qs,[force(q) for q in restart_qs],restart_states,eigh)
            for q,r,s,t in zip(qs,restart_qs,states,restart_states):
                assert torch.equal(q,r) and torch.equal(s.hessian,t.hessian)
                assert torch.equal(s.previous_positions,t.previous_positions)
                assert torch.equal(s.previous_forces,t.previous_forces) and s.step_count==t.step_count
        if step==3:
            buffer=io.BytesIO()
            torch.save([s.state_dict() for s in states],buffer)
            buffer.seek(0)
            snapshots=torch.load(buffer,weights_only=True)
            restart_states=[ASEBFGSState.from_state_dict(s) for s in snapshots]
            restart_qs=[q.clone() for q in qs]
    # Same candidate on the original single-system math body.
    single_q=qs[0].clone()
    single_state=ASEBFGSState.from_state_dict(states[0].state_dict())
    reference_q=[q.clone() for q in qs]
    reference_states=[ASEBFGSState.from_state_dict(s.state_dict()) for s in states]
    _step_with_eigh(single_q,force(single_q),single_state,eigh)
    _step_batch_with_eigh(reference_q,[force(q) for q in reference_q],reference_states,eigh)
    torch.testing.assert_close(single_q,reference_q[0],atol=1e-12,rtol=1e-10)

    def assert_unchanged(q_before, snapshots):
        assert all(torch.equal(q,before) for q,before in zip(qs,q_before))
        for state,expected in zip(states,snapshots):
            actual=state.state_dict()
            for key,value in expected.items():
                assert torch.equal(actual[key],value) if isinstance(value,torch.Tensor) else actual[key]==value

    failures=[]
    for mode in ["zero_spectrum","nonconvergence","invalid_vectors"]:
        if mode=="zero_spectrum":
            states[1]=bfgs_ase_prepare_state(widths[1],device="cuda",system_id=1,alpha=53,maxstep=0.12)
            states[1].hessian.zero_()
        snapshots=[s.state_dict() for s in states]
        before=[q.clone() for q in qs]
        def fail_eigh(a,**kwargs):
            if mode=="nonconvergence":
                w,v,*_=checked_raw(a,max_sweeps=1)
                return w,v
            if mode=="invalid_vectors":
                w,v=eigh(a,**kwargs)
                return w, v*2 if a.shape[-1]==widths[1] else v
            return eigh(a,**kwargs)
        try:
            _step_batch_with_eigh(qs,[force(q) for q in qs],states,fail_eigh)
        except BFGSNumericalError as exc:
            failures.append(dict(mode=mode,error=str(exc)))
        else:
            raise AssertionError(f"{mode} not rejected")
        assert_unchanged(before,snapshots)
        states=[ASEBFGSState.from_state_dict(s.state_dict()) for s in restart_states]
        qs=[q.clone() for q in restart_qs]
    result=dict(status="passed",widths=widths,steps=10,group_shapes=shapes,
                errors_max_abs=errors,restart_bitwise_equal=True,single_batch_gate=True,failures=failures)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "correctness", "bfgs"), required=True)
    parser.add_argument("--source", type=Path, default=Path("artifacts/ase-bfgs-m4-2026-10-09/fault-matrices"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--regression-matrix",type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[147,285,147])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "correctness":
        if args.dataset is None:
            parser.error("--dataset is required for HIP qualification")
        correctness(args)
    else:
        bfgs_gate(args)


if __name__ == "__main__":
    main()
