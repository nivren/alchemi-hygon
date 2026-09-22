"""P11 communication/compute overlap smoke for a DTK PyTorch process group."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist


DEFAULT_PAYLOADS = "1M,16M"


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


def sync_all(device: torch.device, compute_stream: torch.cuda.Stream) -> None:
    torch.cuda.synchronize(device)
    compute_stream.synchronize()


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


def stats(samples_ms: list[float]) -> dict[str, object]:
    ordered = sorted(samples_ms)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.90) - 1))
    mean_ms = statistics.mean(ordered)
    return {
        "samples_ms": [round(value, 6) for value in samples_ms],
        "min_ms": round(min(ordered), 6),
        "median_ms": round(statistics.median(ordered), 6),
        "mean_ms": round(mean_ms, 6),
        "p90_ms": round(ordered[p90_index], 6),
        "max_ms": round(max(ordered), 6),
        "stdev_ms": round(statistics.pstdev(ordered), 6),
        "relative_stdev": round(statistics.pstdev(ordered) / mean_ms, 6),
    }


def run_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    stream: torch.cuda.Stream | None,
) -> None:
    if stream is None:
        torch.mm(left, right, out=output)
    else:
        with torch.cuda.stream(stream):
            torch.mm(left, right, out=output)


def check_compute(output: torch.Tensor, matrix_size: int) -> None:
    expected = torch.full_like(output, float(matrix_size))
    exact_check(output, expected, "matmul.correctness")


def prepare_compute(matrix_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    left = torch.ones((matrix_size, matrix_size), dtype=torch.float32, device=device)
    right = torch.ones_like(left)
    output = torch.empty_like(left)
    return left, right, output


def prepare_comm(count: int, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = torch.full((count,), float(rank + 1), dtype=torch.float32, device=device)
    working = torch.empty_like(baseline)
    return baseline, working


def run_serial(
    baseline: torch.Tensor,
    working: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    expected_comm: torch.Tensor,
    matrix_size: int,
    device: torch.device,
    compute_stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    def operation(check: bool = False) -> None:
        working.copy_(baseline)
        dist.all_reduce(working)
        run_matmul(left, right, output, None)
        sync_all(device, compute_stream)
        if check:
            exact_check(working, expected_comm, "serial.all_reduce.correctness")
            check_compute(output, matrix_size)

    dist.barrier()
    operation(check=True)
    dist.barrier()
    for _ in range(warmup):
        operation()
    dist.barrier()
    samples_ms: list[float] = []
    for _ in range(iterations):
        working.copy_(baseline)
        sync_all(device, compute_stream)
        dist.barrier()
        sync_all(device, compute_stream)
        start = time.perf_counter()
        dist.all_reduce(working)
        run_matmul(left, right, output, None)
        sync_all(device, compute_stream)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        dist.barrier()
    result = stats(samples_ms)
    result.update({"path": "serial", "correctness": "PASS", "stream": "default"})
    return result


def run_overlap(
    baseline: torch.Tensor,
    working: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    expected_comm: torch.Tensor,
    matrix_size: int,
    device: torch.device,
    compute_stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    def operation(check: bool = False) -> None:
        working.copy_(baseline)
        work = dist.all_reduce(working, async_op=True)
        run_matmul(left, right, output, compute_stream)
        work.wait()
        sync_all(device, compute_stream)
        if check:
            exact_check(working, expected_comm, "overlap.all_reduce.correctness")
            check_compute(output, matrix_size)

    dist.barrier()
    operation(check=True)
    dist.barrier()
    for _ in range(warmup):
        operation()
    dist.barrier()
    samples_ms: list[float] = []
    for _ in range(iterations):
        working.copy_(baseline)
        sync_all(device, compute_stream)
        dist.barrier()
        sync_all(device, compute_stream)
        start = time.perf_counter()
        work = dist.all_reduce(working, async_op=True)
        run_matmul(left, right, output, compute_stream)
        work.wait()
        sync_all(device, compute_stream)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        dist.barrier()
    result = stats(samples_ms)
    result.update(
        {
            "path": "overlap",
            "correctness": "PASS",
            "stream": "independent_cuda_stream",
            "async_handle_waited": True,
            "synchronized_completion": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("TORCH_DIST_BACKEND", "nccl"))
    parser.add_argument("--payloads", type=parse_payloads, default=parse_payloads(DEFAULT_PAYLOADS))
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("this probe requires a DTK/HIP PyTorch build")
    if args.matrix_size <= 0 or args.warmup < 1 or args.iterations < 5:
        raise ValueError("matrix-size must be positive, warmup >= 1, iterations >= 5")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise RuntimeError(f"P11 matrix expects world_size 2, 4, or 8; got {world_size}")
    device_count = torch.cuda.device_count()
    if device_count != world_size:
        raise RuntimeError(
            f"expected one visible DCU per rank: device_count={device_count} world_size={world_size}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    compute_stream = torch.cuda.Stream(device=device)
    dist.init_process_group(
        backend=args.backend,
        timeout=dt.timedelta(seconds=args.timeout),
        device_id=local_rank,
    )
    started = time.time()
    results: list[dict[str, object]] = []
    try:
        for payload_bytes in args.payloads:
            count = max(1, math.ceil(payload_bytes / 4))
            actual_payload_bytes = count * 4
            baseline, working = prepare_comm(count, rank, device)
            expected_comm = torch.full(
                (count,), float(world_size * (world_size + 1) // 2), dtype=torch.float32, device=device
            )
            left, right, output = prepare_compute(args.matrix_size, device)
            print(
                json.dumps(
                    {
                        "probe": "P11_overlap",
                        "event": "case_start",
                        "rank": rank,
                        "payload_bytes": payload_bytes,
                        "actual_payload_bytes": actual_payload_bytes,
                        "matrix_size": args.matrix_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            serial = run_serial(
                baseline,
                working,
                left,
                right,
                output,
                expected_comm,
                args.matrix_size,
                device,
                compute_stream,
                args.warmup,
                args.iterations,
            )
            overlap = run_overlap(
                baseline,
                working,
                left,
                right,
                output,
                expected_comm,
                args.matrix_size,
                device,
                compute_stream,
                args.warmup,
                args.iterations,
            )
            serial_median = float(serial["median_ms"])
            overlap_median = float(overlap["median_ms"])
            result = {
                "payload_bytes": payload_bytes,
                "actual_payload_bytes": actual_payload_bytes,
                "matrix_size": args.matrix_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "timing_mode": "synchronized_wall_clock",
                "serial": serial,
                "overlap": overlap,
                "overlap_to_serial": round(overlap_median / serial_median, 6),
                "descriptive_time_reduction": round(1.0 - overlap_median / serial_median, 6),
                "overlap_median_less_than_serial": overlap_median < serial_median,
            }
            results.append(result)
            print(
                json.dumps(
                    {"probe": "P11_overlap", "event": "case_pass", "rank": rank, **result},
                    sort_keys=True,
                ),
                flush=True,
            )

        dist.barrier()
        env_snapshot = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("NCCL_") or key.startswith("RCCL_")
        }
        print(
            json.dumps(
                {
                    "probe": "P11_overlap",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                    "backend_requested": args.backend,
                    "backend_actual": dist.get_backend(),
                    "independent_stream": True,
                    "timing_mode": "synchronized_wall_clock",
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "environment_overrides": env_snapshot,
                    "results": results,
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
