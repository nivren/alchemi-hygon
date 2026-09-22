#!/usr/bin/env python3
"""Minimal P1 single-device tensor smoke for a DTK PyTorch runtime.

Run one process per physical DCU with HIP_VISIBLE_DEVICES set by the caller.
The probe deliberately keeps the contract small: device allocation, H2D/D2H,
elementwise work, floating-point matmul, synchronization, and allocator
release/reuse.  It never falls back to CPU for the device operations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch


def _close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.dtype.is_floating_point:
        return bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5))
    return bool(torch.equal(actual, expected))


def _memory() -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }


def run() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "requested_logical_device": args.device,
        "checks": {},
    }

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    count = torch.cuda.device_count()
    result["device_count"] = count
    if not 0 <= args.device < count:
        raise RuntimeError(f"logical device {args.device} is outside device_count={count}")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    result["device"] = str(device)
    result["device_name"] = torch.cuda.get_device_name(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()

    for name, dtype in (
        ("fp32", torch.float32),
        ("fp64", torch.float64),
        ("int64", torch.int64),
    ):
        host = torch.arange(1024, dtype=dtype)
        x = host.to(device)
        if x.device != device:
            raise RuntimeError(f"{name}: tensor landed on {x.device}, expected {device}")
        torch.cuda.synchronize(device)

        y = x * 2 + 1
        torch.cuda.synchronize(device)
        round_trip = y.cpu()
        expected = host * 2 + 1
        if not _close(round_trip, expected):
            raise RuntimeError(f"{name}: H2D/elementwise/D2H mismatch")
        result["checks"][name] = {
            "allocation": "PASS",
            "h2d": "PASS",
            "elementwise": "PASS",
            "d2h": "PASS",
        }
        del x, y, host, expected, round_trip

    matmul: dict[str, str] = {}
    for name, dtype, rtol, atol in (
        ("fp32", torch.float32, 1e-4, 1e-5),
        ("fp64", torch.float64, 1e-7, 1e-8),
    ):
        host_a = torch.arange(64 * 64, dtype=dtype).reshape(64, 64) / 100
        host_b = torch.flip(host_a, dims=[0])
        a = host_a.to(device)
        b = host_b.to(device)
        c = a @ b
        torch.cuda.synchronize(device)
        actual = c.cpu()
        expected = host_a @ host_b
        if not bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)):
            raise RuntimeError(f"{name}: matmul mismatch")
        matmul[name] = "PASS"
        del host_a, host_b, a, b, c, actual, expected
    result["matmul"] = matmul

    torch.cuda.synchronize(device)
    result["memory_before_release"] = _memory()
    del matmul
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    result["memory_after_release"] = _memory()

    reuse_host = torch.ones(2048, dtype=torch.float32)
    reuse = reuse_host.to(device)
    reuse_result = reuse + 3
    torch.cuda.synchronize(device)
    if not bool(torch.equal(reuse_result.cpu(), reuse_host + 3)):
        raise RuntimeError("allocator reuse: mismatch")
    del reuse_host, reuse, reuse_result
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    result["reuse"] = "PASS"
    result["memory_after_reuse"] = _memory()
    result["status"] = "PASS"
    return result


def main() -> int:
    try:
        result = run()
    except Exception as exc:  # keep one machine-readable failure line
        print(
            "P1_RESULT "
            + json.dumps({"status": "FAIL", "error": repr(exc)}, sort_keys=True),
            flush=True,
        )
        return 1
    print("P1_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
