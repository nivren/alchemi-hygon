"""P10 single-node RCCL/PyTorch communication latency baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import time
from collections.abc import Callable

import torch
import torch.distributed as dist


DEFAULT_PAYLOADS = "1K,4K,64K,1M,16M,64M"


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


def event_timing_available(device: torch.device) -> tuple[bool, str | None]:
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()
        start.elapsed_time(end)
        sync(device)
    except Exception as exc:  # pragma: no cover - depends on the installed DTK runtime
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def time_operation(
    operation: Callable[[], None],
    device: torch.device,
    use_events: bool,
) -> float:
    if use_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        sync(device)
        return elapsed_ms

    sync(device)
    start_time = time.perf_counter()
    operation()
    sync(device)
    return (time.perf_counter() - start_time) * 1000.0


def stats(samples_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples_ms)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.90) - 1))
    median_ms = statistics.median(ordered)
    return {
        "samples_ms": [round(value, 6) for value in samples_ms],
        "min_ms": round(min(ordered), 6),
        "median_ms": round(median_ms, 6),
        "mean_ms": round(statistics.mean(ordered), 6),
        "p90_ms": round(ordered[p90_index], 6),
        "max_ms": round(max(ordered), 6),
        "stdev_ms": round(statistics.pstdev(ordered), 6),
        "relative_stdev": round(statistics.pstdev(ordered) / statistics.mean(ordered), 6),
    }


def effective_gib_per_s(logical_bytes: int, median_ms: float) -> float:
    if median_ms <= 0:
        return 0.0
    return logical_bytes / (median_ms / 1000.0) / (1024**3)


def equal_count(payload_bytes: int, world_size: int, element_size: int = 4) -> tuple[int, int]:
    count = max(1, math.ceil(payload_bytes / element_size))
    count += (-count) % world_size
    return count, count * element_size


def make_alltoall_input(count: int, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    chunk = count // world_size
    chunks = [
        torch.arange(chunk, dtype=torch.float32, device=device)
        + rank * 1_000_000
        + destination * 10_000
        for destination in range(world_size)
    ]
    return torch.cat(chunks)


def expected_alltoall(count: int, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    chunk = count // world_size
    chunks = [
        torch.arange(chunk, dtype=torch.float32, device=device)
        + source * 1_000_000
        + rank * 10_000
        for source in range(world_size)
    ]
    return torch.cat(chunks)


def run_all_reduce(
    count: int,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    use_events: bool,
) -> dict[str, object]:
    baseline = torch.full((count,), float(rank + 1), dtype=torch.float32, device=device)
    working = torch.empty_like(baseline)
    expected = torch.full(
        (count,), float(world_size * (world_size + 1) // 2), dtype=torch.float32, device=device
    )

    def prepare() -> None:
        working.copy_(baseline)

    def operation() -> None:
        dist.all_reduce(working)

    dist.barrier()
    prepare()
    operation()
    sync(device)
    exact_check(working, expected, "all_reduce.correctness")
    dist.barrier()
    for _ in range(warmup):
        prepare()
        operation()
    sync(device)
    dist.barrier()
    samples = []
    for _ in range(iterations):
        prepare()
        samples.append(time_operation(operation, device, use_events))
    dist.barrier()
    result = stats(samples)
    result.update(
        {
            "operation": "all_reduce",
            "correctness": "PASS",
            "logical_bytes": int(count * 4 * 2 * (world_size - 1) / world_size),
        }
    )
    result["effective_gib_per_s_median"] = round(
        effective_gib_per_s(int(result["logical_bytes"]), float(result["median_ms"])), 6
    )
    return result


def run_all_gather(
    count: int,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    use_events: bool,
) -> dict[str, object]:
    source = torch.arange(count, dtype=torch.float32, device=device) + rank * 1_000_000
    output = [torch.empty_like(source) for _ in range(world_size)]
    expected = torch.cat(
        [torch.arange(count, dtype=torch.float32, device=device) + source_rank * 1_000_000
         for source_rank in range(world_size)]
    )

    def operation() -> None:
        dist.all_gather(output, source)

    dist.barrier()
    operation()
    sync(device)
    exact_check(torch.cat(output), expected, "all_gather.correctness")
    dist.barrier()
    for _ in range(warmup):
        operation()
    sync(device)
    dist.barrier()
    samples = [time_operation(operation, device, use_events) for _ in range(iterations)]
    dist.barrier()
    result = stats(samples)
    result.update(
        {
            "operation": "all_gather",
            "correctness": "PASS",
            "logical_bytes": int(count * 4 * (world_size - 1)),
        }
    )
    result["effective_gib_per_s_median"] = round(
        effective_gib_per_s(int(result["logical_bytes"]), float(result["median_ms"])), 6
    )
    return result


def run_all_to_all(
    count: int,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    use_events: bool,
) -> dict[str, object]:
    split = count // world_size
    source = make_alltoall_input(count, rank, world_size, device)
    output = torch.empty_like(source)
    expected = expected_alltoall(count, rank, world_size, device)
    splits = [split] * world_size

    def operation() -> None:
        dist.all_to_all_single(
            output,
            source,
            output_split_sizes=splits,
            input_split_sizes=splits,
        )

    dist.barrier()
    operation()
    sync(device)
    exact_check(output, expected, "all_to_all_single.correctness")
    dist.barrier()
    for _ in range(warmup):
        operation()
    sync(device)
    dist.barrier()
    samples = [time_operation(operation, device, use_events) for _ in range(iterations)]
    dist.barrier()
    result = stats(samples)
    result.update(
        {
            "operation": "all_to_all_single",
            "correctness": "PASS",
            "logical_bytes": int(count * 4 * (world_size - 1) / world_size),
        }
    )
    result["effective_gib_per_s_median"] = round(
        effective_gib_per_s(int(result["logical_bytes"]), float(result["median_ms"])), 6
    )
    return result


def run_p2p_pingpong(
    count: int,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    use_events: bool,
) -> dict[str, object]:
    peer = rank ^ 1
    send = torch.full((count,), float(rank + 1), dtype=torch.float32, device=device)
    receive = torch.empty_like(send)
    expected = torch.full((count,), float(peer + 1), dtype=torch.float32, device=device)

    def phase(*, sending: bool) -> None:
        operation = dist.isend if sending else dist.irecv
        requests = dist.batch_isend_irecv([dist.P2POp(operation, send if sending else receive, peer)])
        for request in requests:
            request.wait()

    def operation() -> None:
        phase(sending=rank % 2 == 0)
        phase(sending=rank % 2 == 1)

    dist.barrier()
    operation()
    sync(device)
    exact_check(receive, expected, "p2p_pingpong.correctness")
    dist.barrier()
    for _ in range(warmup):
        operation()
    sync(device)
    dist.barrier()
    samples = [time_operation(operation, device, use_events) for _ in range(iterations)]
    dist.barrier()
    result = stats(samples)
    result.update(
        {
            "operation": "p2p_pingpong",
            "peer": peer,
            "correctness": "PASS",
            "logical_bytes": int(count * 4 * 2),
        }
    )
    result["effective_gib_per_s_median"] = round(
        effective_gib_per_s(int(result["logical_bytes"]), float(result["median_ms"])), 6
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=parse_payloads, default=parse_payloads(DEFAULT_PAYLOADS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 20:
        raise ValueError("P10 requires warmup >= 1 and iterations >= 20")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise ValueError(f"P10 matrix expects world_size 2, 4, or 8; got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    visible_count = torch.cuda.device_count()
    if visible_count != world_size:
        raise RuntimeError(
            f"expected one visible DCU per rank: device_count={visible_count} world_size={world_size}"
        )

    event_ok, event_error = event_timing_available(device)
    init_start = time.perf_counter()
    dist.init_process_group(backend=args.backend, device_id=local_rank)
    init_elapsed_s = time.perf_counter() - init_start
    try:
        results = []
        runners = {
            "all_reduce": run_all_reduce,
            "all_gather": run_all_gather,
            "all_to_all_single": run_all_to_all,
            "p2p_pingpong": run_p2p_pingpong,
        }
        for payload_bytes in args.payloads:
            count, actual_payload_bytes = equal_count(payload_bytes, world_size)
            for operation_name, runner in runners.items():
                print(
                    json.dumps(
                        {
                            "probe": "P10_performance",
                            "event": "case_start",
                            "rank": rank,
                            "operation": operation_name,
                            "requested_payload_bytes": payload_bytes,
                            "actual_payload_bytes": actual_payload_bytes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                result = runner(
                    count,
                    rank,
                    world_size,
                    device,
                    args.warmup,
                    args.iterations,
                    event_ok,
                )
                result.update(
                    {
                        "requested_payload_bytes": payload_bytes,
                        "actual_payload_bytes": actual_payload_bytes,
                        "warmup": args.warmup,
                        "iterations": args.iterations,
                        "timing_mode": "cuda_event" if event_ok else "synced_wall_clock",
                    }
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            "probe": "P10_performance",
                            "event": "case_pass",
                            "rank": rank,
                            **result,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        env_snapshot = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("NCCL_") or key.startswith("RCCL_")
        }
        print(
            json.dumps(
                {
                    "probe": "P10_performance",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "backend_requested": args.backend,
                    "backend_actual": str(dist.get_backend()),
                    "device": torch.cuda.get_device_name(local_rank),
                    "timing_mode": "cuda_event" if event_ok else "synced_wall_clock",
                    "timing_mode_error": event_error,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "init_elapsed_s": round(init_elapsed_s, 6),
                    "environment_overrides": env_snapshot,
                    "results": results,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
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
