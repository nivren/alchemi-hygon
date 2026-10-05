# SPDX-License-Identifier: Apache-2.0
"""Bounded native cell-tools CPU/HCU parity, including heterogeneous empty blocks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from nvalchemiops.backend import resolve_backend
from nvalchemiops.executor import execute_selected

from nvalchemi.dynamics._ops.cell_align import align_cell


def case(dtype, device, counts):
    b, n = len(counts), sum(counts)
    h = torch.tensor(
        [[[2.0, 0.3, -0.2], [0.5, 2.7, 0.1], [0.3, -0.4, 3.1]]],
        dtype=dtype,
        device=device,
    ).repeat(b, 1, 1)
    if b:
        h[-1] *= -1  # left-handed regression
    idx = torch.repeat_interleave(
        torch.arange(b, device=device),
        torch.tensor(counts, dtype=torch.int64, device=device),
    )
    frac = torch.arange(n * 3, device=device, dtype=dtype).reshape(n, 3) / 17
    r = torch.einsum("ni,nij->nj", frac, h[idx])
    original_h = h.clone()
    selected = resolve_backend(
        "torch_reference",
        operation="cell_tools",
        device=device,
        dtype=dtype,
        gradient_order=0,
        features={"row_cell"},
    )

    def no_legacy(*args, **kwargs):
        raise AssertionError("Torch reference must not call Warp")

    def call(name, *args, **kwargs):
        return execute_selected(selected, name, no_legacy, *args, **kwargs)

    tol = 2e-5 if dtype == torch.float32 else 2e-12
    transform = torch.empty_like(h)
    call("align_cell", r, h, transform, idx)
    torch.testing.assert_close(h @ h.mT, original_h @ original_h.mT, rtol=tol, atol=tol)
    actual_frac = torch.linalg.solve(h[idx].mT, r.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual_frac, frac, rtol=tol, atol=tol)
    # Also invoke the existing framework API on the already aligned cells.
    align_cell(r, h, idx, selection=selected)
    ptr = torch.tensor([0] + counts, dtype=torch.int64, device=device).cumsum(0)
    ext_ptr = torch.empty_like(ptr)
    call("extend_atom_ptr", ptr, ext_ptr)
    appended = torch.empty(n + 2 * b, dtype=idx.dtype, device=device)
    call("extend_batch_idx", idx, n, b, appended)
    assert torch.equal(
        appended, torch.cat((idx, torch.arange(b, device=device).repeat_interleave(2)))
    )
    ext = torch.empty(n + 2 * b, 3, dtype=dtype, device=device)
    unpacked_r, unpacked_h = torch.empty_like(r), torch.empty_like(h)
    snapshots = {
        "positions": r,
        "cell": h,
        "transform": transform,
        "ptr": ext_ptr,
        "appended_idx": appended,
    }
    for pack, unpack in [
        ("pack_positions_with_cell", "unpack_positions_with_cell"),
        ("pack_velocities_with_cell", "unpack_velocities_with_cell"),
    ]:
        call(pack, r, h, ext, ptr, ext_ptr, batch_idx=idx)
        call(
            unpack,
            ext,
            unpacked_r,
            unpacked_h,
            atom_ptr=ptr,
            ext_atom_ptr=ext_ptr,
            batch_idx=idx,
        )
        assert torch.equal(unpacked_r, r) and torch.equal(unpacked_h, h)
        snapshots[pack] = ext.clone()
    call("pack_forces_with_cell", r, h, ext, ptr, ext_ptr, batch_idx=idx)
    snapshots["forces"] = ext.clone()
    masses = torch.arange(n, dtype=dtype, device=device) + 1
    cell_masses = torch.arange(b, dtype=dtype, device=device) + 10
    packed_mass = torch.empty(n + 2 * b, dtype=dtype, device=device)
    call("pack_masses_with_cell", masses, cell_masses, packed_mass, ptr, ext_ptr)
    snapshots["masses"] = packed_mass
    owner = torch.repeat_interleave(torch.arange(b, device=device), ptr.diff() + 2)
    expected_mass = torch.empty_like(packed_mass)
    atomic_rows = torch.arange(n, device=device) + 2 * idx
    cell_rows = ext_ptr[:-1] + ptr.diff()
    expected_mass[atomic_rows] = masses
    expected_mass[cell_rows] = cell_masses
    expected_mass[cell_rows + 1] = cell_masses
    assert torch.equal(expected_mass, packed_mass)
    assert torch.equal(owner[atomic_rows], idx)
    assert torch.equal(owner[cell_rows], torch.arange(b, device=device))
    # Expected failure remains on the actual target device, before mutation.
    bad = h.clone()
    if b:
        bad[0].zero_()
        try:
            call("align_cell", r.clone(), bad, torch.empty_like(bad), idx)
        except ValueError:
            pass
        else:
            raise AssertionError("singular cell accepted")
    # Host transfer is evidence collection only, after all operation calls.
    return {k: v.cpu() for k, v in snapshots.items()}, {
        "dtype": str(dtype),
        "counts": counts,
        "num_atoms": n,
        "fractional_max_abs_error": float((actual_frac - frac).abs().max())
        if n
        else 0.0,
        "selection": selected.as_dict(),
        "roundtrip_exact": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("HCU unavailable; no CPU fallback")
    cases = []
    for dtype in (torch.float64, torch.float32):
        for counts in ([2, 0, 3], [0], []):
            expected, _ = case(dtype, torch.device("cpu"), counts)
            actual, evidence = case(dtype, device, counts)
            tol = 2e-5 if dtype == torch.float32 else 2e-12
            error = 0.0
            for key, value in actual.items():
                torch.testing.assert_close(value, expected[key], rtol=tol, atol=tol)
                if value.numel() and value.is_floating_point():
                    error = max(error, float((value - expected[key]).abs().max()))
            evidence["cpu_parity_max_abs_error"] = error
            cases.append(evidence)
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(device),
        "physical_visible_devices": os.getenv("HIP_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cases": cases,
        "scope": "cell primitives only; no MPA-0 or optimizer steps",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
