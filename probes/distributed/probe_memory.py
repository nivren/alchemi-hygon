"""P13 allocator and device-memory behavior probe for large collectives."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist


OPERATIONS = ("all_gather", "all_to_all")
RELEASE_TOLERANCE_BYTES = 8 * 1024 * 1024


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


def memory_snapshot(device: torch.device) -> dict[str, int]:
    sync(device)
    return {
        "memory_allocated": int(torch.cuda.memory_allocated(device)),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "memory_reserved": int(torch.cuda.memory_reserved(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def exact_fill(
    actual: torch.Tensor,
    expected_value: float,
    scratch: torch.Tensor,
    label: str,
) -> None:
    torch.eq(actual, expected_value, out=scratch)
    if not bool(torch.all(scratch).item()):
        mismatch = (~scratch).flatten().nonzero(as_tuple=False)
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        actual_value = actual.flatten()[first].item() if first >= 0 else "n/a"
        raise AssertionError(
            f"{label}: mismatch at flat index {first}; "
            f"actual={actual_value} expected={expected_value}"
        )


def aligned_elements(payload_bytes: int, world_size: int) -> tuple[int, int]:
    elements = max(1, math.ceil(payload_bytes / 4))
    elements += (-elements) % world_size
    return elements, elements * 4


def make_all_gather_runner(
    rank: int,
    world_size: int,
    device: torch.device,
    elements: int,
) -> Callable[[], None]:
    source = torch.full((elements,), float(rank + 1), dtype=torch.float32, device=device)
    gathered = [torch.empty_like(source) for _ in range(world_size)]
    scratch = torch.empty((elements,), dtype=torch.bool, device=device)

    def run() -> None:
        dist.all_gather(gathered, source)
        sync(device)
        for source_rank, output in enumerate(gathered):
            exact_fill(output, float(source_rank + 1), scratch, f"all_gather[{source_rank}]")

    return run


def make_all_to_all_runner(
    rank: int,
    world_size: int,
    device: torch.device,
    elements: int,
) -> Callable[[], None]:
    source = torch.empty((elements * world_size,), dtype=torch.float32, device=device)
    chunk = elements
    for destination in range(world_size):
        source[destination * chunk : (destination + 1) * chunk].fill_(float(rank + 1))
    output = torch.empty_like(source)
    scratch = torch.empty((chunk,), dtype=torch.bool, device=device)
    splits = [chunk] * world_size
    def run() -> None:
        dist.all_to_all_single(
            output,
            source,
            output_split_sizes=splits,
            input_split_sizes=splits,
        )
        sync(device)
        for source_rank in range(world_size):
            output_chunk = output[source_rank * chunk : (source_rank + 1) * chunk]
            exact_fill(output_chunk, float(source_rank + 1), scratch, f"all_to_all[{source_rank}]")

    return run


def stage(
    operation: str,
    rank: int,
    world_size: int,
    device: torch.device,
    payload_bytes: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    elements, actual_payload_bytes = aligned_elements(payload_bytes, world_size)
    baseline = memory_snapshot(device)
    dist.barrier()

    if operation == "all_gather":
        runner = make_all_gather_runner(rank, world_size, device, elements)
    elif operation == "all_to_all":
        runner = make_all_to_all_runner(rank, world_size, device, elements)
    else:
        raise ValueError(f"unknown operation: {operation}")

    for _ in range(warmup):
        runner()
    dist.barrier()
    prepared = memory_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeats):
        runner()
    peak = memory_snapshot(device)
    elapsed_s = time.perf_counter() - started
    dist.barrier()

    # Keep the two release observations separate: PyTorch may release allocated
    # blocks while retaining them in its cache until empty_cache() is requested.
    del runner
    sync(device)
    gc.collect()
    sync(device)
    after_release = memory_snapshot(device)
    torch.cuda.empty_cache()
    sync(device)
    after_empty_cache = memory_snapshot(device)

    persistent_delta = max(
        after_empty_cache["memory_allocated"] - baseline["memory_allocated"], 0
    )
    return {
        "operation": operation,
        "requested_payload_bytes": payload_bytes,
        "actual_payload_bytes": actual_payload_bytes,
        "warmup": warmup,
        "repeats": repeats,
        "baseline": baseline,
        "prepared": prepared,
        "peak_during_repeats": peak,
        "after_release": after_release,
        "after_empty_cache": after_empty_cache,
        "peak_allocated_delta": peak["max_memory_allocated"] - prepared["memory_allocated"],
        "peak_reserved_delta": peak["max_memory_reserved"] - prepared["memory_reserved"],
        "persistent_allocated_delta": persistent_delta,
        "release_check": (
            "PASS" if persistent_delta <= RELEASE_TOLERANCE_BYTES else "FAIL"
        ),
        "elapsed_s": round(elapsed_s, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("TORCH_DIST_BACKEND", "nccl"))
    parser.add_argument("--payload", type=parse_payload, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("this probe requires a DTK/HIP PyTorch build")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("--warmup and --repeats must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise RuntimeError(f"P13 matrix expects world_size 2, 4, or 8; got {world_size}")
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
        # Establish the communicator and allocator baseline before the large
        # buffers are created. This keeps first-use setup out of P13 peaks.
        warmup = torch.ones(1, dtype=torch.float32, device=device)
        dist.all_reduce(warmup)
        sync(device)
        del warmup
        gc.collect()
        sync(device)
        dist.barrier()
        initial_memory = memory_snapshot(device)
        print(
            json.dumps(
                {
                    "probe": "P13_memory",
                    "event": "start",
                    "rank": rank,
                    "world_size": world_size,
                    "payload_bytes": args.payload,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "operations": list(OPERATIONS),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        for operation in OPERATIONS:
            print(
                json.dumps(
                    {
                        "probe": "P13_memory",
                        "event": "stage_start",
                        "rank": rank,
                        "operation": operation,
                        "payload_bytes": args.payload,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            result = stage(
                operation,
                rank,
                world_size,
                device,
                args.payload,
                args.warmup,
                args.repeats,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "probe": "P13_memory",
                        "event": "stage_pass",
                        "rank": rank,
                        **result,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        status = "PASS" if all(item["release_check"] == "PASS" for item in results) else "FAIL"
        dist.barrier()
        print(
            json.dumps(
                {
                    "probe": "P13_memory",
                    "status": status,
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                    "backend_requested": args.backend,
                    "backend_actual": dist.get_backend(),
                    "initial_memory": initial_memory,
                    "results": results,
                    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "elapsed_s": round(time.time() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if status == "PASS" else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
