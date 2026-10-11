# SPDX-License-Identifier: Apache-2.0
"""M5-4 registered solver compared with unchanged probe serial and rocSOLVER."""

import argparse
import json
import time
from pathlib import Path

import torch
from ase_bfgs_heterogeneous import Scheduler
from ase_bfgs_heterogeneous_math import step
from ase_bfgs_jacobi_candidate_benchmark import stats
from ase_bfgs_jacobi_candidate_test import validate
from nvalchemiops import _bfgs_jacobi_hip as product
from nvalchemiops._bfgs_eigh_hip import _eigh
from nvalchemiops.bfgs_ase import bfgs_ase_prepare_state
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh as product_math

VARIANTS = ("rocsolver", "serial", "product")


def measure(functions, repeats, reset=lambda name: None):
    cold = {}
    for name, fn in functions.items():
        reset(name)
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        cold[name] = (time.perf_counter() - start) * 1000
    for _ in range(2):
        for name, fn in functions.items():
            reset(name)
            fn()
    torch.cuda.synchronize()
    wall = {k: [] for k in functions}
    events = {k: [] for k in functions}
    for repeat in range(repeats):
        order = list(functions)
        # Alternating forward/reverse ordering, cyclic starting offset.
        offset = (repeat // 2) % len(order)
        order = order[offset:] + order[:offset]
        if repeat % 2:
            order.reverse()
        for name in order:
            reset(name)
            torch.cuda.synchronize()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start = time.perf_counter()
            begin.record()
            functions[name]()
            end.record()
            end.synchronize()
            wall[name].append((time.perf_counter() - start) * 1000)
            events[name].append(begin.elapsed_time(end))
    return dict(
        first_call_ms=cold,
        wall_ms={k: stats(v) for k, v in wall.items()},
        caller_event_ms={k: stats(v) for k, v in events.items()},
        warmup=2,
        repeats=repeats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-oracles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", choices=VARIANTS)
    parser.add_argument("--input-set", choices=("li", "t2", "mixed"), default="li")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--full-math", action="store_true")
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error("seven samples required")
    torch.set_num_threads(1)
    real = torch.load(args.real_oracles, map_location="cpu", weights_only=True)
    schedulers = {v: Scheduler(v) for v in VARIANTS if v != "product"}
    schedulers["product"] = product._grouped_eigh
    result = dict(
        status="running",
        torch=str(torch.__version__),
        hip=torch.version.hip,
        device=torch.cuda.get_device_name(),
        cases=[],
    )
    try:
        for scope in (
            [args.input_set]
            if args.trace
            else [
                "li",
                "t2",
                "mixed",
                "synthetic147B1",
                "synthetic189B32",
                "synthetic561B8",
            ]
        ):
            refs = [r for r in real if scope == "mixed" or r["scope"] == scope]
            if scope.startswith("synthetic"):
                n, batch = {
                    "synthetic147B1": (147, 1),
                    "synthetic189B32": (189, 32),
                    "synthetic561B8": (561, 8),
                }[scope]
                gen = torch.Generator().manual_seed(5400 + n)
                u = torch.randn(batch, n, 18, generator=gen, dtype=torch.float64)
                h = 70 * torch.eye(n, dtype=torch.float64)[None] - 0.03 * (u @ u.mT)
                references = []
                for matrix in h:
                    w, v = torch.linalg.eigh(matrix)
                    references.append(
                        dict(
                            input=matrix,
                            symmetric=matrix,
                            w=w,
                            v=v,
                            n=n,
                            name="synthetic_rank18",
                        )
                    )
                refs = [dict(scope=scope, input=h, references=references)]
            hs = [r["input"].cuda() for r in refs]
            row = dict(
                scope=scope,
                shapes=[list(h.shape) for h in hs],
                samples_scope="checked full eigensystem",
            )
            functions = {v: lambda v=v: schedulers[v](hs) for v in VARIANTS}
            if args.trace:
                fn = functions[args.trace]
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                begin = time.time_ns()
                out = fn()
                torch.cuda.synchronize()
                end = time.time_ns()
                row.update(
                    variant=args.trace,
                    window_unix_ns=[begin, end],
                    correctness=[
                        validate(h, o, r["references"])
                        for h, o, r in zip(hs, out, refs)
                    ],
                )
                result["cases"].append(row)
                break
            row["correctness"] = {}
            row["peak_extra_torch_allocator_bytes"] = {}
            for variant, fn in functions.items():
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = fn()
                torch.cuda.synchronize()
                row["peak_extra_torch_allocator_bytes"][variant] = (
                    torch.cuda.max_memory_allocated() - before
                )
                row["correctness"][variant] = [
                    validate(h, o, r["references"]) for h, o, r in zip(hs, out, refs)
                ]
                del out
            row["checked_solver"] = measure(functions, args.repeats)
            if args.full_math:
                flat_h = [matrix for h in hs for matrix in h]
                pools = {}
                for variant in VARIANTS:
                    qs = [
                        torch.linspace(
                            -0.02, 0.03, h.shape[-1], device="cuda", dtype=torch.float64
                        ).reshape(-1, 3)
                        for h in flat_h
                    ]
                    states = [
                        bfgs_ase_prepare_state(q.numel(), device="cuda", system_id=i)
                        for i, q in enumerate(qs)
                    ]
                    pools[variant] = (
                        qs,
                        [-q for q in qs],
                        states,
                        [q.clone() for q in qs],
                    )

                def reset(variant):
                    qs, forces, states, initial = pools[variant]
                    for q, f, state, start, h in zip(
                        qs, forces, states, initial, flat_h
                    ):
                        q.copy_(start)
                        state.hessian.copy_(h)
                        state.previous_positions.copy_(q - 0.001)
                        state.previous_forces.copy_(f + 0.002)
                        state.initialized = True
                        state.step_count = 3

                def math(variant):
                    q, f, s, _ = pools[variant]
                    if variant == "product":
                        return product_math(
                            q, f, s, product._eigh, grouped_eigh=product._grouped_eigh
                        )
                    return (
                        _step_batch_with_eigh(q, f, s, _eigh)
                        if variant == "rocsolver"
                        else step(q, f, s, schedulers[variant])
                    )

                row["complete_math"] = measure(
                    {v: lambda v=v: math(v) for v in VARIANTS}, args.repeats, reset
                )
                for variant in VARIANTS[1:]:
                    for q, reference in zip(pools[variant][0], pools["rocsolver"][0]):
                        torch.testing.assert_close(q, reference, atol=1e-10, rtol=1e-8)
                row["complete_math_scope"] = (
                    "captured real H plus synthetic q/f/history; Hessian update, solver, guards, direction, commit; not MLIP workflow"
                )
            result["cases"].append(row)
            print(
                json.dumps(
                    dict(
                        scope=scope,
                        solver_ms={
                            v: d["median"]
                            for v, d in row["checked_solver"]["wall_ms"].items()
                        },
                        math_ms={
                            v: d["median"]
                            for v, d in row.get("complete_math", {})
                            .get("wall_ms", {})
                            .items()
                        },
                    )
                ),
                flush=True,
            )
    finally:
        for name, scheduler in schedulers.items():
            if name != "product":
                scheduler.close()
        product._close_schedulers()
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.trace:
        args.output.with_suffix(".result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
