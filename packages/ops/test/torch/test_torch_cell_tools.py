# SPDX-License-Identifier: Apache-2.0
"""Independent geometry/layout oracles for native six-DOF cell utilities."""

import numpy as np
import pytest
import torch
from ase.cell import Cell

from nvalchemiops import cell_reference as tools


def inputs(dtype, device="cpu"):
    h = torch.tensor(
        [
            [[2.0, 0.3, -0.2], [0.5, 2.7, 0.1], [0.3, -0.4, 3.1]],
            [[-3.0, 0.2, 0.1], [0.2, 2.0, -0.3], [0.4, 0.1, 2.4]],
        ],
        dtype=dtype,
        device=device,
    )
    idx = torch.tensor([0, 0, 1, 1, 1], device=device)
    frac = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.8, -0.1, 1.2],
            [0.4, 0.7, 0.2],
            [-0.3, 0.1, 0.5],
            [0.2, 0.4, -0.2],
        ],
        dtype=dtype,
        device=device,
    )
    return torch.einsum("ni,nij->nj", frac, h[idx]), h, idx, frac


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_alignment_geometry_left_right_and_strides(dtype):
    r, h, idx, frac = inputs(dtype)
    old_r, old_h = r.clone(), h.clone()
    # Noncontiguous destinations must receive the updates.
    r = torch.stack((r, r), -1)[..., 0]
    h = torch.stack((h, h), -1)[..., 0]
    transform = torch.empty_like(h)
    result = tools.align_cell(r, h, transform, idx)
    assert result[0] is r and result[1] is h
    tol = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(h @ h.mT, old_h @ old_h.mT, rtol=tol, atol=tol)
    torch.testing.assert_close(
        torch.linalg.solve(h[idx].mT, r[..., None])[..., 0], frac, rtol=tol, atol=tol
    )
    torch.testing.assert_close(
        transform @ transform.mT,
        torch.eye(3, dtype=dtype).expand_as(h),
        rtol=tol,
        atol=tol,
    )
    torch.testing.assert_close(h @ transform, old_h, rtol=tol, atol=tol)
    # Reproduce the locked upstream negative-volume transform bug: flipping
    # H before computing T maps original fractional coordinates to -s.
    old_transform = torch.linalg.solve(h[1], -old_h[1])
    old_rotated = old_r[2:] @ old_transform.mT
    old_fractional = old_rotated @ torch.linalg.inv(h[1])
    torch.testing.assert_close(old_fractional, -frac[2:], rtol=tol, atol=tol)
    assert (old_fractional - frac[2:]).abs().max() > 1
    assert torch.count_nonzero(torch.triu(h, 1)) == 0
    assert (torch.linalg.det(h) > 0).all()
    # ASE keeps input handedness; negate its left-handed lower form to
    # compare with upstream's positive-diagonal canonical form.
    for b in range(2):
        standard, _ = Cell(old_h[b].numpy()).standard_form("lower")
        expected = np.asarray(standard) * (
            1 if np.linalg.det(old_h[b].numpy()) > 0 else -1
        )
        torch.testing.assert_close(
            h[b], torch.tensor(expected, dtype=dtype), rtol=tol, atol=tol
        )
    shift = torch.tensor([[1.0, -1.0, 2.0]], dtype=dtype)
    before = old_r[1] - old_r[0] + shift @ old_h[0]
    after = r[1] - r[0] + shift @ h[0]
    torch.testing.assert_close(
        after.square().sum(), before.square().sum(), rtol=tol, atol=tol
    )
    # A radial harmonic bond energy is invariant under the same mapping.
    torch.testing.assert_close(
        0.5 * after.square().sum(), 0.5 * before.square().sum(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("counts", [[2, 0, 3], [0], []])
def test_pack_roundtrip_and_exact_interleaved_ownership(dtype, counts):
    b, n = len(counts), sum(counts)
    ptr = torch.tensor([0] + list(np.cumsum(counts)), dtype=torch.int64)
    ext_ptr = torch.empty_like(ptr)
    assert tools.extend_atom_ptr(ptr, ext_ptr) is ext_ptr
    idx = torch.repeat_interleave(
        torch.arange(b), torch.tensor(counts, dtype=torch.int64)
    )
    r = torch.arange(n * 3, dtype=dtype).reshape(n, 3)
    h = torch.tril(torch.arange(b * 9, dtype=dtype).reshape(b, 3, 3))
    ext = torch.empty(n + 2 * b, 3, dtype=dtype)
    for pack, unpack in [
        (tools.pack_positions_with_cell, tools.unpack_positions_with_cell),
        (tools.pack_velocities_with_cell, tools.unpack_velocities_with_cell),
    ]:
        assert pack(r, h, ext, ptr, ext_ptr, batch_idx=idx) is ext
        restored_r, restored_h = torch.empty_like(r), torch.empty_like(h)
        unpack(
            ext,
            restored_r,
            restored_h,
            atom_ptr=ptr,
            ext_atom_ptr=ext_ptr,
            batch_idx=idx,
        )
        assert torch.equal(restored_r, r) and torch.equal(restored_h, h)
    assert tools.pack_forces_with_cell(r, h, ext, ptr, ext_ptr, batch_idx=idx) is ext
    ownership = torch.repeat_interleave(
        torch.arange(b), torch.tensor(counts, dtype=torch.int64) + 2
    )
    for s, count in enumerate(counts):
        start = int(ext_ptr[s])
        assert torch.equal(ext[start : start + count], r[int(ptr[s]) : int(ptr[s + 1])])
        assert torch.equal(ext[start + count], h[s, :, 0])
        assert torch.equal(ext[start + count + 1], h[s, [1, 2, 2], [1, 1, 2]])
        assert (ownership[start : start + count + 2] == s).all()
    mass, cell_mass = torch.arange(n, dtype=dtype), torch.arange(b, dtype=dtype) + 10
    out = torch.empty(n + 2 * b, dtype=dtype)
    tools.pack_masses_with_cell(mass, cell_mass, out, ptr, ext_ptr)
    for s, count in enumerate(counts):
        start = int(ext_ptr[s])
        assert torch.equal(
            out[start : start + count], mass[int(ptr[s]) : int(ptr[s + 1])]
        )
        assert torch.equal(
            out[start + count : start + count + 2], cell_mass[s].expand(2)
        )
    appended = torch.empty(n + 2 * b, dtype=idx.dtype)
    tools.extend_batch_idx(idx, n, b, appended)
    assert torch.equal(appended, torch.cat((idx, torch.arange(b).repeat_interleave(2))))
    if counts == [2, 0, 3]:
        assert not torch.equal(appended, ownership)  # two distinct upstream layouts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_system_projection_and_required_num_atoms(dtype):
    r, h, _, _ = inputs(dtype)
    h = h[:1].clone()
    out = torch.empty(7, 3, dtype=dtype)
    tools.pack_positions_with_cell(r, h, out)
    restored_r, restored_h = torch.empty_like(r), torch.empty_like(h)
    with pytest.raises(ValueError, match="num_atoms"):
        tools.unpack_positions_with_cell(out, restored_r, restored_h)
    tools.unpack_positions_with_cell(out, restored_r, restored_h, 5)
    assert torch.equal(restored_r, r)
    assert torch.equal(restored_h, torch.tril(h))


@pytest.mark.parametrize("bad", ["singular", "nan", "idx", "dtype", "grad", "alias"])
def test_alignment_errors_before_mutation(bad):
    r, h, idx, _ = inputs(torch.float64)
    t = torch.empty_like(h)
    if bad == "singular":
        h[0].zero_()
    elif bad == "nan":
        h[0, 0, 0] = float("nan")
    elif bad == "idx":
        idx[0] = 2
    elif bad == "dtype":
        t = t.float()
    elif bad == "grad":
        r.requires_grad_()
    else:
        t = h
    before = r.clone()
    with pytest.raises((ValueError, TypeError)):
        tools.align_cell(r, h, t, idx)
    assert torch.equal(r, before)


@pytest.mark.parametrize("bad", ["ptr", "ext_ptr", "idx", "capacity", "alias"])
def test_pack_invalid_metadata_and_buffers(bad):
    r, h, idx, _ = inputs(torch.float64)
    ptr = torch.tensor([0, 2, 5])
    ext_ptr = torch.tensor([0, 4, 9])
    out = torch.empty(9, 3, dtype=r.dtype)
    if bad == "ptr":
        ptr[1] = 6
    elif bad == "ext_ptr":
        ext_ptr[1] = 3
    elif bad == "idx":
        idx[2] = 0
    elif bad == "capacity":
        out = out[:8]
    else:
        out = torch.empty(9, 3, dtype=r.dtype)
        r = out[:5]
    with pytest.raises((ValueError, TypeError)):
        tools.pack_positions_with_cell(r, h, out, ptr, ext_ptr, batch_idx=idx)


def test_empty_alignment_and_single_zero_atoms():
    for b in (0, 1):
        r = torch.empty(0, 3, dtype=torch.float64)
        h = torch.eye(3, dtype=r.dtype).repeat(b, 1, 1)
        before = h.clone()
        tools.align_cell(r, h, torch.empty_like(h))
        assert torch.equal(h, before)


def test_strided_pack_outputs_and_integer_overflow():
    r, h, _, _ = inputs(torch.float64)
    ext = torch.empty(7, 3, 2, dtype=r.dtype)[..., 0]
    tools.pack_positions_with_cell(r, h[:1], ext)
    out_r = torch.empty(5, 3, 2, dtype=r.dtype)[..., 0]
    out_h = torch.empty(1, 3, 3, 2, dtype=r.dtype)[..., 0]
    tools.unpack_positions_with_cell(ext, out_r, out_h, 5)
    assert torch.equal(out_r, r) and torch.equal(out_h, torch.tril(h[:1]))
    ptr = torch.tensor([0, torch.iinfo(torch.int32).max], dtype=torch.int32)
    with pytest.raises(ValueError, match="overflow"):
        tools.extend_atom_ptr(ptr, torch.empty_like(ptr))


@pytest.mark.parametrize(
    "bad", ["float16", "float_idx", "negative_ptr", "device", "nonfinite_mass"]
)
def test_explicit_invalid_inputs(bad):
    r, h, idx, _ = inputs(torch.float64)
    if bad == "float16":
        with pytest.raises(TypeError):
            tools.align_cell(r.half(), h.half(), torch.empty_like(h).half(), idx)
    elif bad == "float_idx":
        with pytest.raises(TypeError):
            tools.align_cell(r, h, torch.empty_like(h), idx.float())
    elif bad == "negative_ptr":
        with pytest.raises(ValueError):
            tools.extend_atom_ptr(
                torch.tensor([0, -1]), torch.empty(2, dtype=torch.int64)
            )
    elif bad == "device":
        with pytest.raises(ValueError, match="device"):
            tools.align_cell(r, h, torch.empty_like(h), idx, device="cuda:0")
    else:
        with pytest.raises(ValueError, match="finite"):
            tools.pack_masses_with_cell(
                torch.tensor([float("nan")]), torch.ones(1), torch.empty(3)
            )
