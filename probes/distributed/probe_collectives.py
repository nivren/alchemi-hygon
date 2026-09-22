"""P4 collective correctness probe for a DTK PyTorch process group.

Run with torchrun on an already allocated set of DCUs.  The probe keeps all
payloads on the selected device and reports one JSON summary per rank.
"""

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


def parse_payloads(value: str) -> list[int]:
    payloads = []
    for item in value.split(","):
        item = item.strip().lower()
        units = {"k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024}
        multiplier = next((factor for suffix, factor in units.items() if item.endswith(suffix)), 1)
        number = item[:-1] if multiplier != 1 else item
        payloads.append(int(float(number) * multiplier))
    if not payloads or any(payload <= 0 for payload in payloads):
        raise argparse.ArgumentTypeError("payloads must be positive byte sizes")
    return payloads


def elements_for(payload_bytes: int, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return max(1, payload_bytes // element_size)


def synchronize(device: torch.device) -> None:
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


def run_timed(
    label: str,
    fn: Callable[[], None],
    device: torch.device,
) -> float:
    dist.barrier()
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    dist.barrier()
    return elapsed_ms


def filled(size: int, value: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.full((size,), value, dtype=dtype, device=device)


def run_dtype_payload(
    dtype_name: str,
    dtype: torch.dtype,
    payload_bytes: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    size = elements_for(payload_bytes, dtype)
    results: list[dict[str, object]] = []

    def record(label: str, fn: Callable[[], None]) -> None:
        elapsed_ms = run_timed(label, fn, device)
        results.append(
            {
                "operation": label,
                "dtype": dtype_name,
                "payload_bytes": payload_bytes,
                "elements": size,
                "elapsed_ms": round(elapsed_ms, 3),
                "status": "PASS",
            }
        )

    record("broadcast", lambda: _broadcast(size, dtype, rank, device))
    record("all_reduce", lambda: _all_reduce(size, dtype, rank, world_size, device))
    record("all_gather", lambda: _all_gather(size, dtype, rank, world_size, device))
    record(
        "all_gather_into_tensor",
        lambda: _all_gather_into_tensor(size, dtype, rank, world_size, device),
    )
    record(
        "all_to_all_single_equal",
        lambda: _all_to_all_equal(size, dtype, rank, world_size, device),
    )
    record(
        "all_to_all_single_variable_zero",
        lambda: _all_to_all_variable(size, dtype, rank, world_size, device),
    )

    if hasattr(dist, "reduce_scatter_tensor"):
        record(
            "reduce_scatter_tensor",
            lambda: _reduce_scatter(size, dtype, rank, world_size, device),
        )
    else:
        results.append(
            {
                "operation": "reduce_scatter_tensor",
                "dtype": dtype_name,
                "payload_bytes": payload_bytes,
                "elements": size,
                "status": "UNSUPPORTED",
            }
        )
    return results


def _broadcast(size: int, dtype: torch.dtype, rank: int, device: torch.device) -> None:
    tensor = torch.arange(size, dtype=dtype, device=device) + 17
    dist.broadcast(tensor, src=0)
    expected = torch.arange(size, dtype=dtype, device=device) + 17
    exact_check(tensor, expected, "broadcast")


def _all_reduce(
    size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    tensor = filled(size, rank + 1, dtype, device)
    dist.all_reduce(tensor)
    expected = filled(size, world_size * (world_size + 1) // 2, dtype, device)
    exact_check(tensor, expected, "all_reduce")


def _all_gather(
    size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    source = filled(size, rank + 1, dtype, device)
    gathered = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(gathered, source)
    for source_rank, tensor in enumerate(gathered):
        exact_check(tensor, filled(size, source_rank + 1, dtype, device), "all_gather")


def _all_gather_into_tensor(
    size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    source = filled(size, rank + 1, dtype, device)
    gathered = torch.empty(size * world_size, dtype=dtype, device=device)
    dist.all_gather_into_tensor(gathered, source)
    expected = torch.cat(
        [filled(size, source_rank + 1, dtype, device) for source_rank in range(world_size)]
    )
    exact_check(gathered, expected, "all_gather_into_tensor")


def _all_to_all_equal(
    size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    per_destination = max(1, size // world_size)
    actual_size = per_destination * world_size
    pieces = [
        filled(per_destination, rank * 100000 + destination * 1000, dtype, device)
        + torch.arange(per_destination, dtype=dtype, device=device)
        for destination in range(world_size)
    ]
    source = torch.cat(pieces)
    output = torch.empty_like(source)
    splits = [per_destination] * world_size
    dist.all_to_all_single(output, source, splits, splits)
    expected = torch.cat(
        [
            filled(per_destination, source_rank * 100000 + rank * 1000, dtype, device)
            + torch.arange(per_destination, dtype=dtype, device=device)
            for source_rank in range(world_size)
        ]
    )
    assert output.numel() == actual_size
    exact_check(output, expected, "all_to_all_single_equal")


def _all_to_all_variable(
    size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    per_destination = max(1, size // max(1, world_size))
    send_splits = [(rank + destination) % 3 for destination in range(world_size)]
    if sum(send_splits) == 0:
        send_splits[0] = 1
    source_chunks = [
        filled(send_count, rank * 100000 + destination * 1000, dtype, device)
        + torch.arange(send_count, dtype=dtype, device=device)
        for destination, send_count in enumerate(send_splits)
    ]
    source = torch.cat(source_chunks) if source_chunks else torch.empty(0, dtype=dtype, device=device)
    recv_splits = [(source_rank + rank) % 3 for source_rank in range(world_size)]
    if sum(recv_splits) == 0:
        recv_splits[0] = 1
    output = torch.empty(sum(recv_splits), dtype=dtype, device=device)
    dist.all_to_all_single(output, source, recv_splits, send_splits)
    expected_chunks = [
        filled(recv_count, source_rank * 100000 + rank * 1000, dtype, device)
        + torch.arange(recv_count, dtype=dtype, device=device)
        for source_rank, recv_count in enumerate(recv_splits)
    ]
    expected = (
        torch.cat(expected_chunks)
        if expected_chunks
        else torch.empty(0, dtype=dtype, device=device)
    )
    exact_check(output, expected, "all_to_all_single_variable_zero")


def _reduce_scatter(
    size: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    output = torch.empty(size, dtype=dtype, device=device)
    source = filled(size * world_size, rank + 1, dtype, device)
    dist.reduce_scatter_tensor(output, source)
    expected = filled(size, world_size * (world_size + 1) // 2, dtype, device)
    exact_check(output, expected, "reduce_scatter_tensor")


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
        default=parse_payloads("4K,1M,64M"),
        help="comma-separated byte sizes, e.g. 4K,1M,64M",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("this probe requires a DTK/HIP PyTorch build")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(f"expected at least 2 ranks, got {world_size}")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible device count={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=args.backend,
        timeout=dt.timedelta(seconds=args.timeout),
    )

    started = time.time()
    results: list[dict[str, object]] = []
    try:
        for dtype_name, dtype in DTYPES.items():
            for payload_bytes in args.payloads:
                results.extend(
                    run_dtype_payload(
                        dtype_name,
                        dtype,
                        payload_bytes,
                        rank,
                        world_size,
                        device,
                    )
                )
        dist.barrier()
        print(
            json.dumps(
                {
                    "probe": "P4_collectives",
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
