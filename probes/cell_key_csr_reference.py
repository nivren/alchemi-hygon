#!/usr/bin/env python3
"""Run the shared cell-key CSR count/start contract on CPU or HCU Torch."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemiops._cell_list_abi import build_cell_csr_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but unavailable")
    device = torch.device(args.device)
    result = build_cell_csr_reference(
        torch.tensor([3, 0, 3, 1], dtype=torch.int32, device=device),
        5,
        global_atom_offset=10,
    )
    expected_counts = torch.tensor([1, 1, 0, 2, 0], dtype=torch.int32, device=device)
    expected_starts = torch.tensor([10, 11, 12, 12, 14], dtype=torch.int32, device=device)
    if not torch.equal(result.cell_counts, expected_counts):
        raise AssertionError("cell counts differ from the CSR contract")
    if not torch.equal(result.cell_starts, expected_starts):
        raise AssertionError("cell starts differ from the CSR contract")
    empty = build_cell_csr_reference(
        torch.empty(0, dtype=torch.int32, device=device), 3, global_atom_offset=7
    )
    if not torch.equal(
        empty.cell_starts,
        torch.tensor([7, 7, 7], dtype=torch.int32, device=device),
    ):
        raise AssertionError("empty CSR starts differ from the contract")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                "counts": [1, 1, 0, 2, 0],
                "starts": [10, 11, 12, 12, 14],
                "empty_offset_starts": [7, 7, 7],
                "performance_measured": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
