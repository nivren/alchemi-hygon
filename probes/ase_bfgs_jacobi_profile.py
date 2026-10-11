# SPDX-License-Identifier: Apache-2.0
"""M5-1 isolated measurements of the existing BFGS Jacobi implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from nvalchemiops._bfgs_eigh_hip import _eigh, _load_extension
from nvalchemiops.bfgs_ase import bfgs_ase_batch_step, bfgs_ase_prepare_state
from nvalchemiops.bfgs_ase_reference import _check_eigensystem


def stats(values):
    return dict(samples=values, median=statistics.median(values),
                mean=statistics.mean(values), minimum=min(values), maximum=max(values),
                stdev=statistics.pstdev(values))


def prepare(source, output):
    """CPU oracle generation is separate from all device profiling."""
    cases, hashes = {}, {}
    for n in (147, 285):
        path = source / f"last-{n}.pt"
        a = torch.load(path, map_location="cpu", weights_only=True).reshape(1, n, n)
        h = a.tril() + a.tril(-1).mT
        w, v = torch.linalg.eigh(h)
        cases[n] = dict(input=a, symmetric=h, w=w, v=v)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch.save(cases, output)
    output.with_suffix(".inputs.json").write_text(json.dumps(hashes, indent=2) + "\n")


def measure(fn, warmup, repeats, reset=lambda: None):
    """Wall and stream elapsed measurements; initialization/restore excluded."""
    reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()
    wall, stream = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        stream.append(begin.elapsed_time(end))
    return dict(first_call_ms=first, wall_ms=stats(wall), stream_elapsed_ms=stats(stream))


def validate(a, w, v, info, target):
    """CPU oracle and device equation/orthogonality checks outside timed regions."""
    assert bool((info == 0).all())
    reference = target["w"].expand(a.shape[0], -1)
    torch.testing.assert_close(w.cpu(), reference, atol=2e-10, rtol=2e-11)
    h = a.tril() + a.tril(-1).mT
    eq = ((h @ v - v * w.unsqueeze(-2)).norm(dim=(-1, -2)) /
          h.norm(dim=(-1, -2)).clamp_min(1)).max().item()
    ort = ((v.mT @ v - torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)).norm(dim=(-1, -2)) /
           a.shape[-1]**0.5).max().item()
    assert eq < 1e-10 and ort < 1e-10
    return dict(eigenvalue_abs_error=(w.cpu() - reference).abs().max().item(),
                equation_relative=eq, orthogonality_relative=ort, info=info.tolist())


def group_functions(cases):
    widths = [147, 285, 147]
    states = [bfgs_ase_prepare_state(n, device="cuda", system_id=i, backend="hip")
              for i, n in enumerate(widths)]
    initial = [torch.linspace(-0.1, 0.2, n, device="cuda", dtype=torch.float64).reshape(-1, 3)
               for n in widths]
    q = [x.clone() for x in initial]
    f = [-x.clone() for x in initial]

    def reset():
        for n, state, x, start in zip(widths, states, q, initial):
            x.copy_(start)
            state.hessian.copy_(cases[n]["input"][0].to("cuda"))
            state.previous_positions.zero_()
            state.previous_forces.zero_()
            state.initialized, state.step_count = False, 0

    def step():
        return bfgs_ase_batch_step(q, f, states, backend="hip")

    return widths, states, initial, q, f, reset, step


def run(args):
    cases = torch.load(args.dataset, map_location="cpu", weights_only=True)
    torch.cuda.set_device(0)
    start = time.perf_counter()
    extension = _load_extension()
    load_ms = (time.perf_counter() - start) * 1000
    result = dict(status="passed", torch=torch.__version__,
                  torch_hcu=getattr(torch.version, "__hcu_version__", None),
                  hip=torch.version.hip, rocsolver=extension.rocsolver_version(),
                  device_name=torch.cuda.get_device_name(), mode=args.mode,
                  extension_load_ms=load_ms, warmup=args.warmup, repeats=args.repeats,
                  batch4_scope="identical copies of captured Hessian", cases=[])
    if args.mode == "micro":
        for n in (147, 285):
            for batch in (1, 4):
                a = cases[n]["input"].repeat(batch, 1, 1).cuda()
                before = a.clone()
                raw = lambda: extension.eigh(a, "L", 100)
                torch.cuda.synchronize()
                first_start = time.perf_counter()
                w, v, info, residual, sweeps = raw()
                torch.cuda.synchronize()
                first_solver_ms = (time.perf_counter() - first_start) * 1000
                correctness = validate(a, w, v, info, cases[n])
                functions = {
                    "raw_native": raw,
                    "checked_eigh": lambda: _eigh(a),
                    "input_check": lambda: bool(torch.isfinite(torch.tril(a)).all()),
                    "output_check": lambda: bool((info != 0).any()) or bool(
                        torch.isfinite(w).all() & torch.isfinite(v).all() & torch.isfinite(residual).all()),
                    "input_clone": lambda: a.clone(),
                    "output_layout": lambda: v.mT.contiguous(),
                    "eigensystem_guard": lambda: _check_eigensystem(a, w, v),
                }
                measurements = {name: measure(fn, args.warmup, args.repeats)
                                for name, fn in functions.items()}
                checked_w, checked_v = _eigh(a)
                validate(a, checked_w, checked_v, info, cases[n])
                assert torch.equal(a, before)
                row = dict(n=n, batch=batch, algorithm="jacobi", correctness=correctness,
                           first_solver_call_ms=first_solver_ms,
                           sweeps=sweeps.tolist(), library_residual=residual.tolist(),
                           measurements=measurements)
                result["cases"].append(row)
                print(json.dumps({"n": n, "batch": batch, "wall_median_ms": {
                    name: m["wall_ms"]["median"] for name, m in measurements.items()}}), flush=True)
    elif args.mode == "trace":
        a = cases[args.n]["input"].repeat(args.batch, 1, 1).cuda()
        for _ in range(args.warmup):
            extension.eigh(a, "L", 100)
        torch.cuda.synchronize()
        begin = time.time_ns()
        w, v, info, residual, sweeps = extension.eigh(a, "L", 100)
        torch.cuda.synchronize()
        end = time.time_ns()
        # Keep the solver window separate from subsequent CPU oracle checks/copies.
        result.update(window_unix_ns=[begin, end], n=args.n, batch=args.batch,
                      sweeps=sweeps.tolist(), library_residual=residual.tolist())
        result["correctness"] = validate(a, w, v, info, cases[args.n])
        args.output.with_suffix(".maps.txt").write_text(Path("/proc/self/maps").read_text())
    elif args.mode in {"group", "torch-profile"}:
        widths, states, initial, q, f, reset, step = group_functions(cases)
        reset()
        outputs = step()
        errors = []
        for n, x, x0, force, output in zip(widths, q, initial, f, outputs):
            w, v = cases[n]["w"][0], cases[n]["v"][0]
            raw = v @ ((force.cpu().flatten() @ v) / w.abs())
            raw = raw.reshape_as(x0.cpu())
            length = raw.norm(dim=-1).max()
            applied = raw * min(1.0, 0.2 / float(length))
            target = x0.cpu() + applied
            torch.testing.assert_close(x.cpu(), target, atol=1e-10, rtol=1e-8)
            errors.append((x.cpu() - target).abs().max().item())
        result.update(widths=widths, group_shapes=[[2, 147, 147], [1, 285, 285]],
                      coordinate_cpu_oracle_errors=errors)
        if args.mode == "group":
            result["measurement"] = measure(step, args.warmup, args.repeats, reset)
        else:
            for _ in range(args.warmup):
                reset()
                step()
            torch.cuda.synchronize()
            reset()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                step()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
            result["profiler"] = [{"key": x.key, "count": x.count,
                                    "cpu_total_us": x.cpu_time_total,
                                    "device_total_us": x.device_time_total,
                                    "self_cpu_us": x.self_cpu_time_total}
                                   for x in prof.key_averages()]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "mode": args.mode}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "micro", "trace", "group", "torch-profile"), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--source", type=Path, default=Path("artifacts/ase-bfgs-m4-2026-10-09/fault-matrices"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, choices=(147, 285), default=285)
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 7 or (args.mode != "prepare" and args.dataset is None):
        parser.error("nonnegative warmup, >=7 repeats, and dataset required for device modes")
    torch.set_num_threads(1)
    prepare(args.source, args.output) if args.mode == "prepare" else run(args)


if __name__ == "__main__":
    main()
