# SPDX-License-Identifier: Apache-2.0
"""M5-3A CPU oracle and bounded dynamic-size HIP qualification."""

import argparse
import json
from pathlib import Path

import torch

from ase_bfgs_jacobi_candidate_test import validate

DIMENSIONS = (1, 2, 3, 7, 138, 189, 199, 200, 201, 511, 512, 513, 552, 561, 1104, 1113)


def prepare(args):
    coverage = []
    for n in (*DIMENSIONS, 147, 285):
        m = n + n % 2
        visits = set()
        for r in range(m - 1):
            endpoints = set()
            for k in range(m // 2):
                p = 0 if k == 0 else 1 + (k - 1 + r) % (m - 1)
                q = 1 + (m - 2 - k + r) % (m - 1)
                p, q = sorted((p, q))
                if q >= n:
                    continue
                assert p != q and not ({p, q} & endpoints) and (p, q) not in visits
                endpoints.update((p, q))
                visits.add((p, q))
        assert len(visits) == n * (n - 1) // 2
        coverage.append(dict(n=n, pairs=len(visits), rounds=m - 1))
    cases = torch.load(args.original, map_location="cpu", weights_only=True)
    for n in DIMENSIONS:
        gen = torch.Generator().manual_seed(5300 + n)
        r = torch.randn(n, n, generator=gen, dtype=torch.float64)
        h = 70 * torch.eye(n, dtype=torch.float64) + (r + r.T) * 0.1
        w, v = torch.linalg.eigh(h)
        cases.append(dict(name="dynamic_dense", n=n, input=h, symmetric=h, w=w, v=v))
    values = torch.tensor([2., -3., 2., 0., -0., -3.], dtype=torch.float64)
    sorted_values, indices = torch.sort(values, stable=True)
    assert indices.tolist() == [1, 5, 3, 4, 0, 2]
    assert torch.equal(sorted_values, values[indices])
    torch.save(cases, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(
        dict(status="passed", execution="CPU", coverage=coverage,
             stable_sort_equal_keys=True, cases=len(cases)), indent=2) + "\n")
    print(json.dumps(dict(status="passed", cases=len(cases), dimensions=list(DIMENSIONS))), flush=True)


def correctness(args):
    from ase_bfgs_jacobi_candidate import checked_raw, load, raw_eigh
    from nvalchemiops.bfgs_ase import BFGSNumericalError, bfgs_ase_prepare_state
    from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh

    cases = torch.load(args.dataset, map_location="cpu", weights_only=True)
    load()
    rows = []
    for case in cases:
        a = case["input"].cuda()
        before = a.view(torch.int64).clone()
        outputs = checked_raw(a)
        row = dict(n=case["n"], name=case["name"], batch=1, **validate(a, outputs, [case]))
        assert torch.equal(before, a.view(torch.int64))
        assert all(out.device == a.device for out in outputs)
        rows.append(row)
        print(json.dumps(row), flush=True)
    chunks = []
    for n, batch in ((147, 8), (189, 16), (285, 8), (199, 32)):
        refs = [c for c in cases if c["n"] == n and c["name"] != "upper_poison"]
        references = [refs[i % len(refs)] for i in range(batch)]
        a = torch.stack([c["input"] for c in references]).cuda()
        before = a.clone()
        full = checked_raw(a)
        split = checked_raw(a, chunk_size=3)
        assert all(torch.equal(x, y) for x, y in zip(full, split))
        assert torch.equal(a, before)
        row = dict(n=n, batch=batch, chunk_size=3, bitwise_equal=True,
                   **validate(a, full, references))
        chunks.append(row)
        print(json.dumps(row), flush=True)
    ref = next(c for c in cases if c["n"] == 513)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        a = ref["input"].cuda()
        outputs = checked_raw(a)
        event = torch.cuda.Event()
        event.record()
    torch.cuda.current_stream().wait_event(event)
    stream_result = validate(a, outputs, [ref])
    failed = raw_eigh(a, max_sweeps=1)
    assert failed[2].tolist() == [1]
    for chunk in (0, -1, True, 1.5):
        try:
            raw_eigh(a, chunk_size=chunk)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk_size was not rejected")

    # A real final-chunk solver failure after earlier diagonal chunks succeed.
    qs = [torch.full((49, 3), .01, device="cuda", dtype=torch.float64) for _ in range(8)]
    states = [bfgs_ase_prepare_state(147, device="cuda", system_id=i) for i in range(8)]
    states[-1].hessian.copy_(next(c["input"] for c in cases
                                if c["n"] == 147 and c["name"] == "random").cuda())
    saved_q = [q.clone() for q in qs]
    saved_state = [s.state_dict() for s in states]
    def failing(h, **kwargs):
        w, v, *_ = checked_raw(h, max_sweeps=1, chunk_size=3)
        return w, v
    try:
        _step_batch_with_eigh(qs, [-q for q in qs], states, failing)
    except BFGSNumericalError as exc:
        failure = str(exc)
    else:
        raise AssertionError("late chunk nonconvergence was not rejected")
    assert all(torch.equal(q, old) for q, old in zip(qs, saved_q))
    for state, old in zip(states, saved_state):
        for key, value in old.items():
            actual = state.state_dict()[key]
            assert torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
    result = dict(status="passed", cases=rows, chunk_cases=chunks,
                  nondefault_stream=stream_result, actual_nonconvergence=True,
                  late_chunk_failure_atomic=True, late_chunk_error=failure,
                  torch=str(torch.__version__), hip=torch.version.hip,
                  device=torch.cuda.get_device_name())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="passed", cases=len(rows), chunk_cases=len(chunks))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "correctness"), required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "prepare":
        prepare(args)
    else:
        correctness(args)


if __name__ == "__main__":
    main()
