"""P8 functional-collectives all-to-all correctness probe for DTK PyTorch."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as functional_collectives


DTYPES = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "int64": torch.int64,
}


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


def split_pattern(direction: str, source: int, destination: int) -> int:
    if direction == "forward":
        return (source + destination) % 3
    return (2 * source + destination + 1) % 4


def make_splits(
    direction: str,
    rank: int,
    world_size: int,
    unit: int,
) -> tuple[list[int], list[int]]:
    input_splits = [
        split_pattern(direction, rank, destination) * unit for destination in range(world_size)
    ]
    if sum(input_splits) == 0:
        input_splits[0] = unit
    output_splits = [
        split_pattern(direction, source, rank) * unit for source in range(world_size)
    ]
    if sum(output_splits) == 0:
        output_splits[0] = unit
    return input_splits, output_splits


def make_input(
    direction: str,
    rank: int,
    input_splits: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for destination, count in enumerate(input_splits):
        base = rank * 1_000_000 + destination * 10_000
        if direction == "reverse":
            base += 100
        chunks.append(
            torch.arange(count, dtype=dtype, device=device) + base
        )
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=dtype, device=device)


def make_expected(
    direction: str,
    rank: int,
    output_splits: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for source, count in enumerate(output_splits):
        base = source * 1_000_000 + rank * 10_000
        if direction == "reverse":
            base += 100
        chunks.append(
            torch.arange(count, dtype=dtype, device=device) + base
        )
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=dtype, device=device)


def functional_exchange(
    direction: str,
    base_count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, object]:
    unit = max(1, base_count // world_size)
    input_splits, output_splits = make_splits(direction, rank, world_size, unit)
    source = make_input(direction, rank, input_splits, dtype, device)

    # This is intentionally the functional primitive, not dist.all_to_all_single.
    result = functional_collectives.all_to_all_single(
        source,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
        tag=f"p8_{direction}",
    )
    result = functional_collectives.wait_tensor(result)
    sync(device)
    expected = make_expected(direction, rank, output_splits, dtype, device)
    exact_check(result, expected, f"functional_all_to_all_single.{direction}")
    return {
        "input_elements": int(source.numel()),
        "output_elements": int(result.numel()),
        "input_splits": input_splits,
        "output_splits": output_splits,
    }


def run_timed(label: str, fn: Callable[[], dict[str, object]], device: torch.device) -> tuple[float, dict[str, object]]:
    dist.barrier()
    sync(device)
    start = time.perf_counter()
    details = fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    dist.barrier()
    return elapsed_ms, details


def run_case(
    direction: str,
    dtype_name: str,
    dtype: torch.dtype,
    payload_bytes: int,
    base_count: int,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, object]:
    label = f"functional_all_to_all_single_{direction}"
    print(
        json.dumps(
            {
                "probe": "P8_functional_collectives",
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
    elapsed_ms, details = run_timed(
        label,
        lambda: functional_exchange(
            direction,
            base_count,
            dtype,
            rank,
            world_size,
            group,
            device,
        ),
        device,
    )
    result = {
        "operation": label,
        "dtype": dtype_name,
        "payload_bytes": payload_bytes,
        "base_elements": base_count,
        "elapsed_ms": round(elapsed_ms, 3),
        "status": "PASS",
        **details,
    }
    print(
        json.dumps(
            {
                "probe": "P8_functional_collectives",
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
    return result


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
        help="comma-separated base byte sizes, e.g. 4K,1M",
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
        if not hasattr(functional_collectives, "all_to_all_single"):
            raise RuntimeError("functional collectives all_to_all_single is unavailable")
        if not hasattr(functional_collectives, "wait_tensor"):
            raise RuntimeError("functional collectives wait_tensor is unavailable")

        group = dist.group.WORLD
        for dtype_name, dtype in DTYPES.items():
            for payload_bytes in args.payloads:
                base_count = elements_for(payload_bytes, dtype)
                for direction in ("forward", "reverse"):
                    results.append(
                        run_case(
                            direction,
                            dtype_name,
                            dtype,
                            payload_bytes,
                            base_count,
                            rank,
                            world_size,
                            group,
                            device,
                        )
                    )

        dist.barrier()
        print(
            json.dumps(
                {
                    "probe": "P8_functional_collectives",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                    "backend_requested": args.backend,
                    "backend_actual": dist.get_backend(),
                    "functional_module": functional_collectives.__file__,
                    "functional_primitives": ["all_to_all_single", "wait_tensor"],
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
