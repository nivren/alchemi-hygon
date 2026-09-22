"""P6 asynchronous collective correctness probe for a DTK PyTorch process group."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Callable

import torch
import torch.distributed as dist


DTYPES = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "int64": torch.int64,
}

CASE_NAMES = (
    "async_all_reduce",
    "async_independent_work",
    "async_outstanding_forward",
    "async_outstanding_reverse",
    "async_repeated",
)


def parse_payloads(value: str) -> list[int]:
    units = {"k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024}
    payloads = []
    for raw in value.split(","):
        item = raw.strip().lower()
        multiplier = next((factor for suffix, factor in units.items() if item.endswith(suffix)), 1)
        number = item[:-1] if multiplier != 1 else item
        payloads.append(int(float(number) * multiplier))
    if not payloads or any(payload <= 0 for payload in payloads):
        raise argparse.ArgumentTypeError("payloads must be positive byte sizes")
    return payloads


def elements_for(payload_bytes: int, dtype: torch.dtype) -> int:
    return max(1, payload_bytes // torch.empty((), dtype=dtype).element_size())


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def exact_check(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).flatten().nonzero(as_tuple=False)
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        raise AssertionError(
            f"{label}: mismatch at flat index {first}; "
            f"actual={actual.flatten()[first].item() if first >= 0 else 'n/a'} "
            f"expected={expected.flatten()[first].item() if first >= 0 else 'n/a'}"
        )


def expected_sum(world_size: int) -> int:
    return world_size * (world_size + 1) // 2


def rank_tensor(
    count: int,
    dtype: torch.dtype,
    rank: int,
    device: torch.device,
    iteration: int = 0,
) -> torch.Tensor:
    value = rank + 1 + iteration * 1000
    return torch.full((count,), value, dtype=dtype, device=device)


def expected_tensor(
    count: int,
    dtype: torch.dtype,
    world_size: int,
    device: torch.device,
    iteration: int = 0,
) -> torch.Tensor:
    value = expected_sum(world_size) + iteration * world_size * 1000
    return torch.full((count,), value, dtype=dtype, device=device)


def async_all_reduce(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    tensor = rank_tensor(count, dtype, rank, device)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    sync(device)
    exact_check(tensor, expected_tensor(count, dtype, world_size, device), "async_all_reduce")


def async_independent_work(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    tensor = rank_tensor(count, dtype, rank, device)
    work = dist.all_reduce(tensor, async_op=True)

    # This tensor is independent of the collective result. It exercises the
    # intended enqueue -> unrelated device work -> wait sequence without
    # reading the collective buffer before completion.
    independent = torch.full((count,), rank + 7, dtype=dtype, device=device)
    independent = independent * 3 + 2

    work.wait()
    sync(device)
    exact_check(tensor, expected_tensor(count, dtype, world_size, device), "async_independent_work")
    exact_check(
        independent,
        torch.full((count,), (rank + 7) * 3 + 2, dtype=dtype, device=device),
        "async_independent_work.independent",
    )


def async_outstanding(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    reverse: bool,
) -> None:
    tensors = [rank_tensor(count, dtype, rank, device, iteration) for iteration in range(3)]
    works = [dist.all_reduce(tensor, async_op=True) for tensor in tensors]

    ordered_works = reversed(works) if reverse else works
    for work in ordered_works:
        work.wait()

    sync(device)
    for iteration, tensor in enumerate(tensors):
        exact_check(
            tensor,
            expected_tensor(count, dtype, world_size, device, iteration),
            f"async_outstanding_{'reverse' if reverse else 'forward'}[{iteration}]",
        )


def async_repeated(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    repeats: int,
) -> None:
    for iteration in range(repeats):
        tensor = rank_tensor(count, dtype, rank, device, iteration)
        work = dist.all_reduce(tensor, async_op=True)
        work.wait()
        sync(device)
        exact_check(
            tensor,
            expected_tensor(count, dtype, world_size, device, iteration),
            f"async_repeated[{iteration}]",
        )


def run_timed(label: str, fn: Callable[[], None], device: torch.device) -> float:
    dist.barrier()
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    dist.barrier()
    return elapsed_ms


def run_matrix(
    dtype_name: str,
    dtype: torch.dtype,
    payload_bytes: int,
    rank: int,
    world_size: int,
    device: torch.device,
    selected_case: str,
) -> list[dict[str, object]]:
    count = elements_for(payload_bytes, dtype)
    cases: list[tuple[str, Callable[[], None]]] = [
        (
            "async_all_reduce",
            lambda: async_all_reduce(count, dtype, rank, world_size, device),
        ),
        (
            "async_independent_work",
            lambda: async_independent_work(count, dtype, rank, world_size, device),
        ),
        (
            "async_outstanding_forward",
            lambda: async_outstanding(count, dtype, rank, world_size, device, False),
        ),
        (
            "async_outstanding_reverse",
            lambda: async_outstanding(count, dtype, rank, world_size, device, True),
        ),
    ]
    if selected_case != "all":
        cases = [case for case in cases if case[0] == selected_case]

    results: list[dict[str, object]] = []
    for label, function in cases:
        print(
            json.dumps(
                {
                    "probe": "P6_async",
                    "event": "case_start",
                    "rank": rank,
                    "operation": label,
                    "dtype": dtype_name,
                    "payload_bytes": payload_bytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        elapsed_ms = run_timed(label, function, device)
        results.append(
            {
                "operation": label,
                "dtype": dtype_name,
                "payload_bytes": payload_bytes,
                "elements": count,
                "elapsed_ms": round(elapsed_ms, 3),
                "status": "PASS",
            }
        )
        print(
            json.dumps(
                {
                    "probe": "P6_async",
                    "event": "case_pass",
                    "rank": rank,
                    "operation": label,
                    "dtype": dtype_name,
                    "payload_bytes": payload_bytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.environ.get("TORCH_DIST_BACKEND", "nccl"),
        help="process-group backend; actual backend is printed after initialization",
    )
    parser.add_argument(
        "--payloads",
        type=parse_payloads,
        default=parse_payloads("4K,1M"),
        help="comma-separated byte sizes, e.g. 4K,1M",
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--case",
        choices=("all", *CASE_NAMES),
        default="all",
        help="run one async case for focused diagnosis, or all cases",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("this probe requires a DTK/HIP PyTorch build")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(f"expected at least 2 ranks, got {world_size}")
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible device count={device_count}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=args.backend,
        timeout=dt.timedelta(seconds=args.timeout),
        device_id=local_rank,
    )
    started = time.time()
    results: list[dict[str, object]] = []
    try:
        for dtype_name, dtype in DTYPES.items():
            if args.case == "async_repeated":
                if dtype_name == "fp32":
                    count = elements_for(args.payloads[0], dtype)
                    label = f"async_repeated_{args.repeats}"
                    print(
                        json.dumps(
                            {
                                "probe": "P6_async",
                                "event": "case_start",
                                "rank": rank,
                                "operation": label,
                                "dtype": dtype_name,
                                "payload_bytes": args.payloads[0],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    elapsed_ms = run_timed(
                        label,
                        lambda: async_repeated(
                            count, dtype, rank, world_size, device, args.repeats
                        ),
                        device,
                    )
                    results.append(
                        {
                            "operation": label,
                            "dtype": dtype_name,
                            "payload_bytes": args.payloads[0],
                            "elements": count,
                            "repeats": args.repeats,
                            "elapsed_ms": round(elapsed_ms, 3),
                            "status": "PASS",
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "probe": "P6_async",
                                "event": "case_pass",
                                "rank": rank,
                                "operation": label,
                                "dtype": dtype_name,
                                "payload_bytes": args.payloads[0],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                continue
            if args.case == "all" or args.case in CASE_NAMES[:4]:
                for payload_bytes in args.payloads:
                    results.extend(
                        run_matrix(
                            dtype_name,
                            dtype,
                            payload_bytes,
                            rank,
                            world_size,
                            device,
                            args.case,
                        )
                    )

        if args.case == "all":
            count = elements_for(args.payloads[0], torch.float32)
            label = f"async_repeated_{args.repeats}"
            print(
                json.dumps(
                    {
                        "probe": "P6_async",
                        "event": "case_start",
                        "rank": rank,
                        "operation": label,
                        "dtype": "fp32",
                        "payload_bytes": args.payloads[0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            elapsed_ms = run_timed(
                label,
                lambda: async_repeated(
                    count, torch.float32, rank, world_size, device, args.repeats
                ),
                device,
            )
            results.append(
                {
                    "operation": label,
                    "dtype": "fp32",
                    "payload_bytes": args.payloads[0],
                    "elements": count,
                    "repeats": args.repeats,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "status": "PASS",
                }
            )
            print(
                json.dumps(
                    {
                        "probe": "P6_async",
                        "event": "case_pass",
                        "rank": rank,
                        "operation": label,
                        "dtype": "fp32",
                        "payload_bytes": args.payloads[0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        dist.barrier()
        print(
            json.dumps(
                {
                    "probe": "P6_async",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                    "backend_requested": args.backend,
                    "backend_actual": dist.get_backend(),
                    "results": results,
                    "elapsed_s": round(time.time() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
