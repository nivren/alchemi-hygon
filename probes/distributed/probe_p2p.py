"""P5 point-to-point correctness probe for a DTK PyTorch process group."""

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
    "pair_isend_irecv",
    "pair_batch_isend_irecv",
    "ring_isend_irecv",
    "ring_variable_asymmetric",
    "bidirectional_pair_batch",
    "zero_length_ring",
    "repeated_small",
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


def exact_check(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).flatten().nonzero(as_tuple=False)
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        raise AssertionError(
            f"{label}: mismatch at flat index {first}; "
            f"actual={actual.flatten()[first].item() if first >= 0 else 'n/a'} "
            f"expected={expected.flatten()[first].item() if first >= 0 else 'n/a'}"
        )


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def message(
    source: int,
    destination: int,
    count: int,
    dtype: torch.dtype,
    device: torch.device,
    tag: int,
) -> torch.Tensor:
    base = source * 100_000 + destination * 1_000 + tag * 10
    return torch.arange(count, dtype=dtype, device=device) + base


def wait_all(requests: list[object]) -> None:
    for request in requests:
        request.wait()


def run_timed(label: str, fn: Callable[[], None], device: torch.device) -> float:
    dist.barrier()
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    dist.barrier()
    return elapsed_ms


def pair(rank: int) -> int:
    return rank ^ 1


def pair_isend_irecv(
    count: int,
    dtype: torch.dtype,
    rank: int,
    device: torch.device,
    tag: int,
) -> None:
    peer = pair(rank)
    lower, upper = sorted((rank, peer))

    # Exercise unbatched isend/irecv without issuing opposite-direction
    # operations in the same order on both ranks. That symmetric pattern is
    # serialized by the eager NCCL-compatible process group and can deadlock
    # before the API contract is actually tested.
    if rank == lower:
        send = message(rank, peer, count, dtype, device, tag)
        dist.isend(send, dst=peer, tag=tag).wait()
    else:
        recv = torch.empty(count, dtype=dtype, device=device)
        dist.irecv(recv, src=peer, tag=tag).wait()
        exact_check(recv, message(peer, rank, count, dtype, device, tag), "isend_irecv.forward")
    dist.barrier()

    if rank == upper:
        send = message(rank, peer, count, dtype, device, tag + 1)
        dist.isend(send, dst=peer, tag=tag + 1).wait()
    else:
        recv = torch.empty(count, dtype=dtype, device=device)
        dist.irecv(recv, src=peer, tag=tag + 1).wait()
        exact_check(recv, message(peer, rank, count, dtype, device, tag + 1), "isend_irecv.reverse")


def pair_batch_isend_irecv(
    count: int,
    dtype: torch.dtype,
    rank: int,
    device: torch.device,
    tag: int,
) -> None:
    peer = pair(rank)
    send = message(rank, peer, count, dtype, device, tag)
    recv = torch.empty(count, dtype=dtype, device=device)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send, peer, tag=tag + rank),
            dist.P2POp(dist.irecv, recv, peer, tag=tag + peer),
        ]
    )
    wait_all(requests)
    exact_check(recv, message(peer, rank, count, dtype, device, tag), "batch_isend_irecv")


def ring_isend_irecv(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    tag: int,
) -> None:
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1) % world_size

    # A ring is a dependency cycle. Split it into two parity phases so the
    # unbatched API is tested without making every rank submit an opposite
    # direction in the same serialized process-group order.
    if rank % 2 == 0:
        send = message(rank, next_rank, count, dtype, device, tag)
        dist.isend(send, dst=next_rank, tag=tag).wait()
    else:
        recv = torch.empty(count, dtype=dtype, device=device)
        dist.irecv(recv, src=previous_rank, tag=tag).wait()
        exact_check(
            recv,
            message(previous_rank, rank, count, dtype, device, tag),
            "ring_isend_irecv.forward",
        )
    dist.barrier()

    if rank % 2 == 1:
        send = message(rank, next_rank, count, dtype, device, tag + 1)
        dist.isend(send, dst=next_rank, tag=tag + 1).wait()
    else:
        recv = torch.empty(count, dtype=dtype, device=device)
        dist.irecv(recv, src=previous_rank, tag=tag + 1).wait()
        exact_check(
            recv,
            message(previous_rank, rank, count, dtype, device, tag + 1),
            "ring_isend_irecv.reverse",
        )


def ring_variable_asymmetric(
    base_count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    tag: int,
) -> None:
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1) % world_size
    increment = max(1, base_count // 8)
    send_count = base_count + rank * increment
    recv_count = base_count + previous_rank * increment
    if rank % 2 == 0:
        send = message(rank, next_rank, send_count, dtype, device, tag)
        dist.isend(send, dst=next_rank, tag=tag).wait()
    else:
        recv = torch.empty(recv_count, dtype=dtype, device=device)
        dist.irecv(recv, src=previous_rank, tag=tag).wait()
        exact_check(
            recv,
            message(previous_rank, rank, recv_count, dtype, device, tag),
            "ring_variable_asymmetric.forward",
        )
    dist.barrier()

    if rank % 2 == 1:
        send = message(rank, next_rank, send_count, dtype, device, tag + 1)
        dist.isend(send, dst=next_rank, tag=tag + 1).wait()
    else:
        recv = torch.empty(recv_count, dtype=dtype, device=device)
        dist.irecv(recv, src=previous_rank, tag=tag + 1).wait()
        exact_check(
            recv,
            message(previous_rank, rank, recv_count, dtype, device, tag + 1),
            "ring_variable_asymmetric.reverse",
        )


def bidirectional_pair_batch(
    count: int,
    dtype: torch.dtype,
    rank: int,
    device: torch.device,
    tag: int,
) -> None:
    peer = pair(rank)
    send_forward = message(rank, peer, count, dtype, device, tag)
    send_reverse = message(rank, peer, count + 1, dtype, device, tag + 100)
    recv_forward = torch.empty(count, dtype=dtype, device=device)
    recv_reverse = torch.empty(count + 1, dtype=dtype, device=device)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send_forward, peer, tag=tag + rank),
            dist.P2POp(dist.irecv, recv_forward, peer, tag=tag + peer),
            dist.P2POp(dist.isend, send_reverse, peer, tag=tag + 100 + rank),
            dist.P2POp(dist.irecv, recv_reverse, peer, tag=tag + 100 + peer),
        ]
    )
    wait_all(requests)
    exact_check(
        recv_forward,
        message(peer, rank, count, dtype, device, tag),
        "bidirectional_pair_batch.forward",
    )
    exact_check(
        recv_reverse,
        message(peer, rank, count + 1, dtype, device, tag + 100),
        "bidirectional_pair_batch.reverse",
    )


def zero_ring(
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    tag: int,
) -> None:
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1) % world_size
    if rank % 2 == 0:
        send = torch.empty(0, dtype=dtype, device=device)
        dist.isend(send, dst=next_rank, tag=tag).wait()
    else:
        recv = torch.empty(0, dtype=dtype, device=device)
        dist.irecv(recv, src=previous_rank, tag=tag).wait()
        assert recv.numel() == 0
    dist.barrier()

    if rank % 2 == 1:
        send = torch.empty(0, dtype=dtype, device=device)
        dist.isend(send, dst=next_rank, tag=tag + 1).wait()
    else:
        recv = torch.empty(0, dtype=dtype, device=device)
        dist.irecv(recv, src=previous_rank, tag=tag + 1).wait()
        assert recv.numel() == 0


def run_cases(
    dtype_name: str,
    dtype: torch.dtype,
    payload_bytes: int,
    rank: int,
    world_size: int,
    device: torch.device,
    selected_case: str,
) -> list[dict[str, object]]:
    count = elements_for(payload_bytes, dtype)
    results: list[dict[str, object]] = []
    cases = [
        (
            "pair_isend_irecv",
            count,
            lambda: pair_isend_irecv(count, dtype, rank, device, 100),
        ),
        (
            "pair_batch_isend_irecv",
            count,
            lambda: pair_batch_isend_irecv(count, dtype, rank, device, 200),
        ),
        (
            "ring_isend_irecv",
            count,
            lambda: ring_isend_irecv(count, dtype, rank, world_size, device, 300),
        ),
        (
            "ring_variable_asymmetric",
            count,
            lambda: ring_variable_asymmetric(count, dtype, rank, world_size, device, 400),
        ),
        (
            "bidirectional_pair_batch",
            count,
            lambda: bidirectional_pair_batch(count, dtype, rank, device, 500),
        ),
    ]
    if selected_case != "all":
        cases = [case for case in cases if case[0] == selected_case]
    for label, elements, function in cases:
        print(
            json.dumps(
                {
                    "probe": "P5_p2p",
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
                "elements": elements,
                "elapsed_ms": round(elapsed_ms, 3),
                "status": "PASS",
            }
        )
        print(
            json.dumps(
                {
                    "probe": "P5_p2p",
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


def run_repeated_small(
    rank: int,
    world_size: int,
    device: torch.device,
    repeats: int,
) -> float:
    count = 4

    def repeat() -> None:
        peer = pair(rank)
        lower, upper = sorted((rank, peer))
        for iteration in range(repeats):
            tag = 10_000 + iteration * 2
            if rank == lower:
                send = message(rank, peer, count, torch.float32, device, tag)
                dist.isend(send, dst=peer, tag=tag).wait()
            else:
                recv = torch.empty(count, dtype=torch.float32, device=device)
                dist.irecv(recv, src=peer, tag=tag).wait()
                exact_check(
                    recv,
                    message(peer, rank, count, torch.float32, device, tag),
                    "repeated_small.forward",
                )

            if rank == upper:
                send = message(rank, peer, count, torch.float32, device, tag + 1)
                dist.isend(send, dst=peer, tag=tag + 1).wait()
            else:
                recv = torch.empty(count, dtype=torch.float32, device=device)
                dist.irecv(recv, src=peer, tag=tag + 1).wait()
                exact_check(
                    recv,
                    message(peer, rank, count, torch.float32, device, tag + 1),
                    "repeated_small.reverse",
                )

    return run_timed(f"repeated_small_{repeats}", repeat, device)


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
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--case",
        choices=("all", *CASE_NAMES),
        default="all",
        help="run one P2P case for focused diagnosis, or all cases",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("this probe requires a DTK/HIP PyTorch build")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or world_size % 2:
        raise RuntimeError(f"P5 requires an even world size >= 2, got {world_size}")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
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
            if args.case in ("all", *CASE_NAMES[:5]):
                for payload_bytes in args.payloads:
                    results.extend(
                        run_cases(
                            dtype_name,
                            dtype,
                            payload_bytes,
                            rank,
                            world_size,
                            device,
                            args.case,
                        )
                    )
            if dtype_name == "fp32" and args.case in ("all", "repeated_small"):
                print(
                    json.dumps(
                        {
                            "probe": "P5_p2p",
                            "event": "case_start",
                            "rank": rank,
                            "operation": f"repeated_small_{args.repeats}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                elapsed_ms = run_repeated_small(rank, world_size, device, args.repeats)
                results.append(
                    {
                        "operation": f"repeated_small_{args.repeats}",
                        "dtype": dtype_name,
                        "payload_bytes": 16,
                        "elements": 4,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "status": "PASS",
                    }
                )
                print(
                    json.dumps(
                        {
                            "probe": "P5_p2p",
                            "event": "case_pass",
                            "rank": rank,
                            "operation": f"repeated_small_{args.repeats}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.case in ("all", "zero_length_ring"):
                print(
                    json.dumps(
                        {
                            "probe": "P5_p2p",
                            "event": "case_start",
                            "rank": rank,
                            "operation": "zero_length_ring",
                            "dtype": dtype_name,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                zero_elapsed_ms = run_timed(
                    "zero_length_ring",
                    lambda: zero_ring(dtype, rank, world_size, device, 800),
                    device,
                )
                results.append(
                    {
                        "operation": "zero_length_ring",
                        "dtype": dtype_name,
                        "payload_bytes": 0,
                        "elements": 0,
                        "elapsed_ms": round(zero_elapsed_ms, 3),
                        "status": "PASS",
                    }
                )
                print(
                    json.dumps(
                        {
                            "probe": "P5_p2p",
                            "event": "case_pass",
                            "rank": rank,
                            "operation": "zero_length_ring",
                            "dtype": dtype_name,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        dist.barrier()
        print(
            json.dumps(
                {
                    "probe": "P5_p2p",
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
