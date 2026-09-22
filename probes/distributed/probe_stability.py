"""P12 long-duration mixed communication stability probe."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


PHASES = (
    ("small", 10_000, 4 * 1024),
    ("medium", 1_000, 1 * 1024 * 1024),
    ("large", 100, 16 * 1024 * 1024),
)
OPERATIONS = (
    "all_reduce",
    "all_gather",
    "all_to_all",
    "p2p_ring",
    "async_all_reduce",
)


def parse_payload(value: str) -> int:
    units = {"k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024}
    item = value.strip().lower()
    multiplier = next((factor for suffix, factor in units.items() if item.endswith(suffix)), 1)
    number = item[:-1] if multiplier != 1 else item
    payload = int(float(number) * multiplier)
    if payload <= 0:
        raise argparse.ArgumentTypeError("payload must be positive")
    return payload


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def exact_check(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).flatten().nonzero(as_tuple=False)
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        actual_value = actual.flatten()[first].item() if first >= 0 else "n/a"
        expected_value = expected.flatten()[first].item() if first >= 0 else "n/a"
        raise AssertionError(
            f"{label}: mismatch at flat index {first}; "
            f"actual={actual_value} expected={expected_value}"
        )


def wait_all(requests: list[object]) -> None:
    for request in requests:
        request.wait()


def memory_snapshot(device: torch.device) -> dict[str, int]:
    sync(device)
    return {
        "memory_allocated": int(torch.cuda.memory_allocated(device)),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "memory_reserved": int(torch.cuda.memory_reserved(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


@dataclass
class PhaseBuffers:
    comm_template: torch.Tensor
    comm: torch.Tensor
    expected_reduce: torch.Tensor
    gather_source_template: torch.Tensor
    gather_source: torch.Tensor
    gather_output: list[torch.Tensor]
    gather_expected: list[torch.Tensor]
    alltoall_source_template: torch.Tensor
    alltoall_source: torch.Tensor
    alltoall_output: torch.Tensor
    alltoall_expected: torch.Tensor
    p2p_send_template: torch.Tensor
    p2p_send: torch.Tensor
    p2p_receive: torch.Tensor
    p2p_expected: torch.Tensor
    alltoall_splits: list[int]


def elements_for(payload_bytes: int, world_size: int) -> tuple[int, int]:
    count = max(1, math.ceil(payload_bytes / 4))
    count += (-count) % world_size
    return count, count * 4


def make_buffers(
    payload_bytes: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> PhaseBuffers:
    count, _ = elements_for(payload_bytes, world_size)
    comm_template = torch.full((count,), float(rank + 1), dtype=torch.float32, device=device)
    comm = torch.empty_like(comm_template)
    expected_reduce = torch.full(
        (count,), float(world_size * (world_size + 1) // 2), dtype=torch.float32, device=device
    )

    gather_source_template = torch.full_like(comm_template, float(rank + 1))
    gather_source = torch.empty_like(gather_source_template)
    gather_output = [torch.empty_like(gather_source) for _ in range(world_size)]
    gather_expected = [
        torch.full_like(gather_source, float(source_rank + 1)) for source_rank in range(world_size)
    ]

    split = count // world_size
    alltoall_source_template = torch.empty_like(comm_template)
    for destination in range(world_size):
        alltoall_source_template[destination * split : (destination + 1) * split].fill_(
            float(rank * 100 + destination)
        )
    alltoall_source = torch.empty_like(alltoall_source_template)
    alltoall_output = torch.empty_like(alltoall_source_template)
    alltoall_expected = torch.empty_like(alltoall_source_template)
    for source_rank in range(world_size):
        alltoall_expected[source_rank * split : (source_rank + 1) * split].fill_(
            float(source_rank * 100 + rank)
        )

    previous_rank = (rank - 1) % world_size
    p2p_send_template = torch.full_like(comm_template, float(rank + 1))
    p2p_send = torch.empty_like(p2p_send_template)
    p2p_receive = torch.empty_like(p2p_send_template)
    p2p_expected = torch.full_like(p2p_send_template, float(previous_rank + 1))

    return PhaseBuffers(
        comm_template=comm_template,
        comm=comm,
        expected_reduce=expected_reduce,
        gather_source_template=gather_source_template,
        gather_source=gather_source,
        gather_output=gather_output,
        gather_expected=gather_expected,
        alltoall_source_template=alltoall_source_template,
        alltoall_source=alltoall_source,
        alltoall_output=alltoall_output,
        alltoall_expected=alltoall_expected,
        p2p_send_template=p2p_send_template,
        p2p_send=p2p_send,
        p2p_receive=p2p_receive,
        p2p_expected=p2p_expected,
        alltoall_splits=[split] * world_size,
    )


def run_operation(
    operation: str,
    buffers: PhaseBuffers,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    if operation in ("all_reduce", "async_all_reduce"):
        buffers.comm.copy_(buffers.comm_template)
        work = dist.all_reduce(buffers.comm, async_op=operation == "async_all_reduce")
        if work is not None:
            work.wait()
        sync(device)
        exact_check(buffers.comm, buffers.expected_reduce, f"{operation}.correctness")
        return

    if operation == "all_gather":
        buffers.gather_source.copy_(buffers.gather_source_template)
        dist.all_gather(buffers.gather_output, buffers.gather_source)
        sync(device)
        for source_rank, (actual, expected) in enumerate(
            zip(buffers.gather_output, buffers.gather_expected)
        ):
            exact_check(actual, expected, f"all_gather[{source_rank}].correctness")
        return

    if operation == "all_to_all":
        buffers.alltoall_source.copy_(buffers.alltoall_source_template)
        dist.all_to_all_single(
            buffers.alltoall_output,
            buffers.alltoall_source,
            output_split_sizes=buffers.alltoall_splits,
            input_split_sizes=buffers.alltoall_splits,
        )
        sync(device)
        exact_check(buffers.alltoall_output, buffers.alltoall_expected, "all_to_all.correctness")
        return

    if operation == "p2p_ring":
        buffers.p2p_send.copy_(buffers.p2p_send_template)
        buffers.p2p_receive.fill_(-1.0)
        next_rank = (rank + 1) % world_size
        previous_rank = (rank - 1) % world_size
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, buffers.p2p_send, next_rank),
                dist.P2POp(dist.irecv, buffers.p2p_receive, previous_rank),
            ]
        )
        wait_all(requests)
        sync(device)
        exact_check(buffers.p2p_receive, buffers.p2p_expected, "p2p_ring.correctness")
        return

    raise ValueError(f"unknown operation: {operation}")


def run_warmup(
    buffers: PhaseBuffers,
    rank: int,
    world_size: int,
    device: torch.device,
    cycles: int,
) -> None:
    for cycle in range(cycles):
        for operation in OPERATIONS:
            run_operation(operation, buffers, rank, world_size, device)
        if cycle + 1 < cycles:
            dist.barrier()
    dist.barrier()


def run_phase(
    phase_name: str,
    total_operations: int,
    payload_bytes: int,
    rank: int,
    world_size: int,
    device: torch.device,
    buffers: PhaseBuffers,
    progress_every: int,
) -> dict[str, object]:
    started = time.perf_counter()
    counts = {operation: 0 for operation in OPERATIONS}
    memory_trace: list[dict[str, object]] = []
    for index in range(total_operations):
        operation = OPERATIONS[index % len(OPERATIONS)]
        run_operation(operation, buffers, rank, world_size, device)
        counts[operation] += 1
        completed = index + 1
        if completed % progress_every == 0 or completed == total_operations:
            memory_trace.append(
                {
                    "completed": completed,
                    "memory": memory_snapshot(device),
                }
            )
            print(
                json.dumps(
                    {
                        "probe": "P12_stability",
                        "event": "progress",
                        "rank": rank,
                        "phase": phase_name,
                        "completed": completed,
                        "total": total_operations,
                        "operation": operation,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier()
    return {
        "phase": phase_name,
        "requested_payload_bytes": payload_bytes,
        "actual_payload_bytes": int(buffers.comm.numel() * 4),
        "operations": total_operations,
        "operation_counts": counts,
        "memory_trace": memory_trace,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("TORCH_DIST_BACKEND", "nccl"))
    parser.add_argument("--small-payload", type=parse_payload, default=4 * 1024)
    parser.add_argument("--medium-payload", type=parse_payload, default=1 * 1024 * 1024)
    parser.add_argument("--large-payload", type=parse_payload, default=16 * 1024 * 1024)
    parser.add_argument("--small-operations", type=int, default=10_000)
    parser.add_argument("--medium-operations", type=int, default=1_000)
    parser.add_argument("--large-operations", type=int, default=100)
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("this probe requires a DTK/HIP PyTorch build")
    if min(args.small_operations, args.medium_operations, args.large_operations) <= 0:
        raise ValueError("phase operation counts must be positive")
    if args.warmup_cycles < 1:
        raise ValueError("--warmup-cycles must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise RuntimeError(f"P12 matrix expects world_size 2, 4, or 8; got {world_size}")
    device_count = torch.cuda.device_count()
    if device_count != world_size:
        raise RuntimeError(
            f"expected one visible DCU per rank: device_count={device_count} world_size={world_size}"
        )

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
        requested_phases = (
            ("small", args.small_operations, args.small_payload),
            ("medium", args.medium_operations, args.medium_payload),
            ("large", args.large_operations, args.large_payload),
        )
        warmup_buffers = make_buffers(args.small_payload, rank, world_size, device)
        print(
            json.dumps(
                {
                    "probe": "P12_stability",
                    "event": "warmup_start",
                    "rank": rank,
                    "cycles": args.warmup_cycles,
                    "operations": list(OPERATIONS),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        run_warmup(warmup_buffers, rank, world_size, device, args.warmup_cycles)
        del warmup_buffers
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = memory_snapshot(device)

        for phase_name, total_operations, payload_bytes in requested_phases:
            buffers = make_buffers(payload_bytes, rank, world_size, device)
            actual_payload_bytes = int(buffers.comm.numel() * 4)
            print(
                json.dumps(
                    {
                        "probe": "P12_stability",
                        "event": "phase_start",
                        "rank": rank,
                        "phase": phase_name,
                        "operations": total_operations,
                        "requested_payload_bytes": payload_bytes,
                        "actual_payload_bytes": actual_payload_bytes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            progress_every = max(1, total_operations // 10)
            results.append(
                run_phase(
                    phase_name,
                    total_operations,
                    payload_bytes,
                    rank,
                    world_size,
                    device,
                    buffers,
                    progress_every,
                )
            )
            print(
                json.dumps(
                    {
                        "probe": "P12_stability",
                        "event": "phase_pass",
                        "rank": rank,
                        **results[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del buffers
            sync(device)

        final_memory = memory_snapshot(device)
        dist.barrier()
        env_snapshot = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("NCCL_") or key.startswith("RCCL_")
        }
        print(
            json.dumps(
                {
                    "probe": "P12_stability",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                    "backend_requested": args.backend,
                    "backend_actual": dist.get_backend(),
                    "operations": list(OPERATIONS),
                    "warmup_cycles": args.warmup_cycles,
                    "initial_memory": initial_memory,
                    "final_memory": final_memory,
                    "results": results,
                    "environment_overrides": env_snapshot,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
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
