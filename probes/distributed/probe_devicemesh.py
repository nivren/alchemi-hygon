"""P7 DeviceMesh and subgroup correctness probe for a DTK PyTorch process group."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


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


def shape_for_2d(world_size: int) -> tuple[int, int]:
    if world_size == 4:
        return (2, 2)
    if world_size == 8:
        return (2, 4)
    raise ValueError(f"P7 2D mesh shape is defined for world size 4 or 8, got {world_size}")


def check_1d_lookup(mesh: DeviceMesh, submesh: DeviceMesh, rank: int, world_size: int) -> None:
    if mesh.ndim != 1 or tuple(mesh.mesh.shape) != (world_size,):
        raise AssertionError(f"unexpected 1D mesh shape: ndim={mesh.ndim}, shape={mesh.mesh.shape}")
    if mesh.mesh.tolist() != list(range(world_size)):
        raise AssertionError(f"unexpected 1D mesh values: {mesh.mesh.tolist()}")
    if mesh.mesh_dim_names != ("world",):
        raise AssertionError(f"unexpected 1D mesh names: {mesh.mesh_dim_names}")
    if mesh.get_coordinate() != [rank]:
        raise AssertionError(f"rank {rank}: unexpected 1D coordinate {mesh.get_coordinate()}")
    if mesh.get_local_rank() != rank or mesh.get_local_rank("world") != rank:
        raise AssertionError(f"rank {rank}: unexpected 1D local rank")
    group = mesh.get_group("world")
    if dist.get_world_size(group) != world_size or dist.get_rank(group) != rank:
        raise AssertionError(f"rank {rank}: unexpected 1D mesh group mapping")
    if submesh.mesh.tolist() != list(range(world_size)):
        raise AssertionError(f"rank {rank}: unexpected 1D submesh values {submesh.mesh.tolist()}")


def check_2d_lookup(
    mesh: DeviceMesh,
    row_submesh: DeviceMesh,
    col_submesh: DeviceMesh,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    rows, cols = shape_for_2d(world_size)
    row, col = divmod(rank, cols)
    expected_mesh = [[index for index in range(start, start + cols)] for start in range(0, world_size, cols)]
    if mesh.ndim != 2 or tuple(mesh.mesh.shape) != (rows, cols):
        raise AssertionError(f"unexpected 2D mesh shape: {mesh.mesh.shape}")
    if mesh.mesh.tolist() != expected_mesh:
        raise AssertionError(f"unexpected 2D mesh values: {mesh.mesh.tolist()}")
    if mesh.mesh_dim_names != ("row", "col"):
        raise AssertionError(f"unexpected 2D mesh names: {mesh.mesh_dim_names}")
    if mesh.get_coordinate() != [row, col]:
        raise AssertionError(f"rank {rank}: unexpected 2D coordinate {mesh.get_coordinate()}")
    if mesh.get_local_rank("row") != row or mesh.get_local_rank("col") != col:
        raise AssertionError(f"rank {rank}: unexpected 2D local rank")

    row_group = mesh.get_group("row")
    col_group = mesh.get_group("col")
    if dist.get_world_size(row_group) != rows or dist.get_rank(row_group) != row:
        raise AssertionError(f"rank {rank}: unexpected row group mapping")
    if dist.get_world_size(col_group) != cols or dist.get_rank(col_group) != col:
        raise AssertionError(f"rank {rank}: unexpected col group mapping")
    if row_submesh.mesh.tolist() != [row_index * cols + col for row_index in range(rows)]:
        raise AssertionError(f"rank {rank}: unexpected row subgroup {row_submesh.mesh.tolist()}")
    if col_submesh.mesh.tolist() != [row * cols + col_index for col_index in range(cols)]:
        raise AssertionError(f"rank {rank}: unexpected col subgroup {col_submesh.mesh.tolist()}")
    if len(mesh.get_all_groups()) != 2:
        raise AssertionError(f"rank {rank}: expected two 2D mesh groups")
    return row, col


def group_all_reduce(
    count: int,
    dtype: torch.dtype,
    rank: int,
    members: list[int],
    group: dist.ProcessGroup,
    device: torch.device,
    label: str,
) -> None:
    tensor = torch.full((count,), rank + 1, dtype=dtype, device=device)
    dist.all_reduce(tensor, group=group)
    sync(device)
    expected_value = sum(member + 1 for member in members)
    expected = torch.full((count,), expected_value, dtype=dtype, device=device)
    exact_check(tensor, expected, label)


def run_timed(label: str, fn: Callable[[], None], device: torch.device) -> float:
    dist.barrier()
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    dist.barrier()
    return elapsed_ms


def run_case(
    label: str,
    dtype_name: str,
    dtype: torch.dtype,
    payload_bytes: int,
    count: int,
    rank: int,
    function: Callable[[], None],
    device: torch.device,
) -> dict[str, object]:
    print(
        json.dumps(
            {
                "probe": "P7_devicemesh",
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
    result = {
        "operation": label,
        "dtype": dtype_name,
        "payload_bytes": payload_bytes,
        "elements": count,
        "elapsed_ms": round(elapsed_ms, 3),
        "status": "PASS",
    }
    print(
        json.dumps(
            {
                "probe": "P7_devicemesh",
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


def run_lookup_case(
    label: str,
    rank: int,
    function: Callable[[], None],
    device: torch.device,
) -> dict[str, object]:
    print(
        json.dumps(
            {
                "probe": "P7_devicemesh",
                "event": "case_start",
                "rank": rank,
                "operation": label,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    elapsed_ms = run_timed(label, function, device)
    result = {"operation": label, "elapsed_ms": round(elapsed_ms, 3), "status": "PASS"}
    print(
        json.dumps(
            {
                "probe": "P7_devicemesh",
                "event": "case_pass",
                "rank": rank,
                "operation": label,
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
        help="comma-separated byte sizes, e.g. 4K,1M",
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
        mesh_1d = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("world",))
        mesh_1d_sub = mesh_1d["world"]
        mesh_1d_group = mesh_1d.get_group("world")
        mesh_1d_subgroup = mesh_1d_sub.get_group()

        results.append(
            run_lookup_case(
                "mesh_1d_rank_lookup",
                rank,
                lambda: check_1d_lookup(mesh_1d, mesh_1d_sub, rank, world_size),
                device,
            )
        )

        mesh_2d = None
        row_submesh = None
        col_submesh = None
        mesh_2d_row_group = None
        mesh_2d_col_group = None
        row, col = 0, 0
        rows, cols = 1, world_size
        if world_size >= 4:
            rows, cols = shape_for_2d(world_size)
            mesh_values = [
                [index for index in range(start, start + cols)]
                for start in range(0, world_size, cols)
            ]
            mesh_2d = DeviceMesh("cuda", mesh_values, mesh_dim_names=("row", "col"))
            row_submesh = mesh_2d["row"]
            col_submesh = mesh_2d["col"]
            mesh_2d_row_group = row_submesh.get_group()
            mesh_2d_col_group = col_submesh.get_group()
            results.append(
                run_lookup_case(
                    "mesh_2d_rank_lookup",
                    rank,
                    lambda: check_2d_lookup(
                        mesh_2d, row_submesh, col_submesh, rank, world_size
                    ),
                    device,
                )
            )
            row, col = divmod(rank, cols)

        for dtype_name, dtype in DTYPES.items():
            for payload_bytes in args.payloads:
                count = elements_for(payload_bytes, dtype)
                results.append(
                    run_case(
                        "mesh_1d_group_all_reduce",
                        dtype_name,
                        dtype,
                        payload_bytes,
                        count,
                        rank,
                        lambda: group_all_reduce(
                            count,
                            dtype,
                            rank,
                            list(range(world_size)),
                            mesh_1d_group,
                            device,
                            "mesh_1d_group_all_reduce",
                        ),
                        device,
                    )
                )
                results.append(
                    run_case(
                        "mesh_1d_subgroup_all_reduce",
                        dtype_name,
                        dtype,
                        payload_bytes,
                        count,
                        rank,
                        lambda: group_all_reduce(
                            count,
                            dtype,
                            rank,
                            list(range(world_size)),
                            mesh_1d_subgroup,
                            device,
                            "mesh_1d_subgroup_all_reduce",
                        ),
                        device,
                    )
                )
                if mesh_2d_row_group is not None and mesh_2d_col_group is not None:
                    row_members = [row_index * cols + col for row_index in range(rows)]
                    col_members = [row * cols + col_index for col_index in range(cols)]
                    results.append(
                        run_case(
                            "mesh_2d_row_subgroup_all_reduce",
                            dtype_name,
                            dtype,
                            payload_bytes,
                            count,
                            rank,
                            lambda: group_all_reduce(
                                count,
                                dtype,
                                rank,
                                row_members,
                                mesh_2d_row_group,
                                device,
                                "mesh_2d_row_subgroup_all_reduce",
                            ),
                            device,
                        )
                    )
                    results.append(
                        run_case(
                            "mesh_2d_col_subgroup_all_reduce",
                            dtype_name,
                            dtype,
                            payload_bytes,
                            count,
                            rank,
                            lambda: group_all_reduce(
                                count,
                                dtype,
                                rank,
                                col_members,
                                mesh_2d_col_group,
                                device,
                                "mesh_2d_col_subgroup_all_reduce",
                            ),
                            device,
                        )
                    )

        dist.barrier()
        print(
            json.dumps(
                {
                    "probe": "P7_devicemesh",
                    "status": "PASS",
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "device": torch.cuda.get_device_name(local_rank),
                    "backend_requested": args.backend,
                    "backend_actual": dist.get_backend(),
                    "mesh_1d": {"shape": [world_size], "dim_names": ["world"]},
                    "mesh_2d": (
                        {"shape": [rows, cols], "dim_names": ["row", "col"]}
                        if mesh_2d is not None
                        else None
                    ),
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
