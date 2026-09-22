"""P9 autograd-sensitive communication smoke for DTK PyTorch."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as functional_collectives


DTYPES = {
    "fp32": torch.float32,
    "fp64": torch.float64,
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
        actual_value = actual.flatten()[first].item() if first >= 0 else "n/a"
        expected_value = expected.flatten()[first].item() if first >= 0 else "n/a"
        raise AssertionError(
            f"{label}: mismatch at flat index {first}; "
            f"actual={actual_value} expected={expected_value}"
        )


def make_local(
    count: int,
    dtype: torch.dtype,
    rank: int,
    device: torch.device,
    requires_grad: bool,
) -> torch.Tensor:
    values = torch.arange(count, dtype=dtype, device=device) + rank * 1_000
    if requires_grad:
        values.requires_grad_()
    return values


def expected_gather(
    count: int,
    dtype: torch.dtype,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(count, dtype=dtype, device=device) + source * 1_000
            for source in range(world_size)
        ]
    )


def native_all_gather_smoke(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Probe direct c10d all_gather without an autograd adapter."""
    local = make_local(count, dtype, rank, device, requires_grad=True)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    sync(device)
    output = torch.cat(gathered)
    exact_check(
        output,
        expected_gather(count, dtype, world_size, device),
        "native_all_gather.forward",
    )

    loss = (output * output).sum()
    native_supported = False
    backward_error = None
    if loss.requires_grad:
        loss.backward()
        if local.grad is None:
            raise AssertionError("native_all_gather: loss had grad path but local.grad is None")
        exact_check(local.grad, 2 * local.detach(), "native_all_gather.backward")
        native_supported = True
    else:
        try:
            loss.backward()
        except RuntimeError as exc:
            backward_error = f"{type(exc).__name__}: {exc}"
        else:
            raise AssertionError("native_all_gather: backward unexpectedly succeeded without grad path")

    return {
        "forward_status": "PASS",
        "native_autograd_supported": native_supported,
        "native_backward_error": backward_error,
        "loss_requires_grad": bool(loss.requires_grad),
    }


def functional_all_gather_autograd_smoke(
    count: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
    tag: str,
) -> dict[str, object]:
    """Probe the explicit functional autograd-enabled gather primitive."""
    local = make_local(count, dtype, rank, device, requires_grad=True)
    output = functional_collectives.all_gather_tensor_autograd(
        local,
        gather_dim=0,
        group=dist.group.WORLD,
        tag=tag,
    )
    output = functional_collectives.wait_tensor(output)
    sync(device)
    exact_check(
        output,
        expected_gather(count, dtype, world_size, device),
        "functional_all_gather_autograd.forward",
    )

    loss = (output * output).sum()
    if not loss.requires_grad:
        raise AssertionError("functional_all_gather_autograd: loss has no grad path")
    loss.backward()
    if local.grad is None:
        raise AssertionError("functional_all_gather_autograd: local.grad is None")
    # Every rank differentiates the same complete gathered loss, so the
    # autograd collective sums world_size identical local contributions.
    exact_check(
        local.grad,
        world_size * 2 * local.detach(),
        "functional_all_gather_autograd.backward",
    )
    return {
        "forward_status": "PASS",
        "backward_status": "PASS",
        "functional_autograd_supported": True,
        "loss_requires_grad": bool(loss.requires_grad),
    }


def run_case(
    dtype_name: str,
    dtype: torch.dtype,
    payload_bytes: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    count = elements_for(payload_bytes, dtype)
    print(
        json.dumps(
            {
                "probe": "P9_autograd_collective",
                "event": "case_start",
                "rank": rank,
                "dtype": dtype_name,
                "payload_bytes": payload_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.barrier()
    sync(device)
    start = time.perf_counter()
    native = native_all_gather_smoke(count, dtype, rank, world_size, device)
    dist.barrier()
    functional = functional_all_gather_autograd_smoke(
        count,
        dtype,
        rank,
        world_size,
        device,
        tag=f"p9_{dtype_name}_{payload_bytes}",
    )
    sync(device)
    dist.barrier()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = {
        "dtype": dtype_name,
        "payload_bytes": payload_bytes,
        "elements_per_rank": count,
        "elapsed_ms": round(elapsed_ms, 3),
        "native": native,
        "functional_autograd": functional,
        "status": "PASS",
    }
    print(
        json.dumps(
            {
                "probe": "P9_autograd_collective",
                "event": "case_pass",
                "rank": rank,
                **result,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=parse_payloads, default=parse_payloads("4K,1M"))
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    visible_count = torch.cuda.device_count()
    if visible_count != world_size:
        raise RuntimeError(
            f"expected one visible DCU per rank: device_count={visible_count} world_size={world_size}"
        )

    dist.init_process_group(backend=args.backend, device_id=local_rank)
    try:
        results = []
        for dtype_name, dtype in DTYPES.items():
            for payload_bytes in args.payloads:
                results.append(
                    run_case(dtype_name, dtype, payload_bytes, rank, world_size, device)
                )

        print(
            json.dumps(
                {
                    "probe": "P9_autograd_collective",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "backend_requested": args.backend,
                    "backend_actual": str(dist.get_backend()),
                    "device": torch.cuda.get_device_name(local_rank),
                    "functional_module": functional_collectives.__file__,
                    "native_autograd_collective_supported": all(
                        item["native"]["native_autograd_supported"] for item in results
                    ),
                    "functional_autograd_supported": all(
                        item["functional_autograd"]["functional_autograd_supported"]
                        for item in results
                    ),
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
