# SPDX-License-Identifier: Apache-2.0
"""Independent per-system NumPy oracle for native variable-cell optimizers."""

import numpy as np
import pytest
import torch

from nvalchemiops.cell_fire_reference import (
    fire2_step_coord_cell,
    fire_cell_post,
    fire_cell_pre,
)


def state(dtype=torch.float64, device="cpu", branch="positive"):
    idx = torch.tensor([0, 0, 1, 1, 1], device=device)
    h = torch.tensor(
        [
            [[2.0, 0, 0], [0.4, 2.5, 0], [-0.2, 0.3, 3.0]],
            [[3.0, 0, 0], [-0.3, 2.0, 0], [0.2, -0.4, 2.4]],
        ],
        dtype=dtype,
        device=device,
    )
    frac = torch.arange(15, dtype=dtype, device=device).reshape(5, 3) / 17
    f = torch.tensor([[0.3, -0.2, 0.1]] * 5, dtype=dtype, device=device)
    fh = torch.tensor(
        [[[0.2, 0, 0], [0.1, -0.3, 0], [0.05, 0.1, 0.4]]] * 2,
        dtype=dtype,
        device=device,
    )
    v = 0.03 * f
    hv = 0.02 * fh
    if branch == "negative":
        v, hv = -10 * f, -10 * fh
    elif branch == "zero":
        f.zero_()
        fh.zero_()
    elif branch == "cell_only":
        f.zero_()
        v.zero_()
    return dict(
        r=torch.einsum("ni,nij->nj", frac, h[idx]),
        v=v,
        f=f,
        h=h,
        hv=hv,
        fh=fh,
        idx=idx,
        a=torch.tensor([0.09, 0.04], dtype=dtype, device=device),
        dt=torch.tensor([0.04, 0.06], dtype=dtype, device=device),
        n=torch.tensor([2, 5], dtype=torch.int32, device=device),
        mass=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype, device=device),
    )


def numpy_state(s):
    return {k: v.cpu().numpy().copy() for k, v in s.items()}


def oracle_fire2(s, maxstep=0.1, delaystep=2, cell_force_scale=1.0):
    out = {k: v.copy() for k, v in s.items()}
    for b in range(len(s["dt"])):
        atoms = np.where(s["idx"] == b)[0]
        v = np.concatenate(
            (s["v"][atoms], [s["hv"][b, :, 0], s["hv"][b, [1, 2, 2], [1, 1, 2]]])
        )
        force = s["fh"][b] / (len(atoms) * cell_force_scale)
        f = np.concatenate((s["f"][atoms], [force[:, 0], force[[1, 2, 2], [1, 1, 2]]]))
        kicked = v + s["dt"][b] * f
        power = np.sum(kicked * f)
        count = s["n"][b] + 1 if power > 0 else 0
        dt, alpha = s["dt"][b], s["a"][b]
        if power <= 0:
            dt, alpha = max(0.005, 0.75 * dt), 0.09
        elif count > delaystep:
            dt, alpha = min(0.08, 1.05 * dt), 0.985 * alpha
        ratio = (
            np.sqrt(np.sum(kicked * kicked) / np.sum(f * f))
            if np.sum(f * f) > 0
            else 0.0
        )
        mixed = (1 - alpha) * v + ((1 - alpha) * s["dt"][b] + alpha * ratio) * f
        vh = np.zeros((3, 3))
        vh[:, 0] = mixed[-2]
        vh[[1, 2, 2], [1, 1, 2]] = mixed[-1]
        factor = 1.0 if power > 0 else -0.5
        dh = factor * dt * vh
        dr = s["r"][atoms] @ np.linalg.inv(s["h"][b]) @ dh + factor * dt * mixed[:-2]
        largest = np.linalg.norm(dr, axis=1).max()
        scale = min(1.0, maxstep / largest) if largest > 0 else 1.0
        out["r"][atoms] += scale * dr
        out["h"][b] += scale * dh
        out["v"][atoms] = mixed[:-2] if power > 0 else 0.0
        out["hv"][b] = vh if power > 0 else 0.0
        out["a"][b] = alpha
        out["dt"][b] = dt * scale
        out["n"][b] = count
    return out


def run_fire2(s, **kwargs):
    fire2_step_coord_cell(
        s["r"],
        s["v"],
        s["f"],
        s["h"],
        s["hv"],
        s["fh"],
        s["idx"],
        s["a"],
        s["dt"],
        s["n"],
        delaystep=2,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("branch", ["positive", "negative", "zero", "cell_only"])
@pytest.mark.parametrize("maxstep", [0.1, 0.0002])
def test_fire2_numpy_single_step(dtype, branch, maxstep):
    s = state(dtype, branch=branch)
    before = numpy_state(s)
    expected = oracle_fire2(before, maxstep=maxstep)
    run_fire2(s, maxstep=maxstep)
    tol = 2e-12 if dtype == torch.float64 else 3e-5
    for key in ("r", "v", "h", "hv", "a", "dt", "n"):
        np.testing.assert_allclose(s[key].numpy(), expected[key], rtol=tol, atol=tol)
    assert torch.equal(s["f"], torch.from_numpy(before["f"]))
    assert torch.equal(s["fh"], torch.from_numpy(before["fh"]))
    assert np.linalg.norm(s["r"].numpy() - before["r"], axis=1).max() <= maxstep + tol


def test_fire2_affine_mapping_is_row_cell_not_old_column_formula():
    s = state(branch="cell_only")
    before = numpy_state(s)
    frac = s["r"] @ torch.linalg.inv(s["h"][0])  # compare separately below
    run_fire2(s, maxstep=1.0)
    for b in range(2):
        mask = s["idx"] == b
        expected = before["r"][mask] @ np.linalg.inv(before["h"][b])
        np.testing.assert_allclose(
            s["r"][mask].numpy() @ np.linalg.inv(s["h"][b].numpy()),
            expected,
            atol=2e-12,
        )
    old_r = before["r"][:2]
    bad = (s["h"][0].numpy() @ np.linalg.inv(before["h"][0]) @ old_r.T).T
    assert np.max(np.abs(bad - s["r"][:2].numpy())) > 1e-7
    del frac


def oracle_fire_pre(s, maxstep):
    out = {k: v.copy() for k, v in s.items()}
    for b in range(2):
        atoms = np.where(s["idx"] == b)[0]
        dt = s["dt"][b]
        out["v"][atoms] += 0.5 * dt * s["f"][atoms] / s["mass"][atoms, None]
        out["hv"][b] += 0.5 * dt * s["fh"][b]
        dh = dt * out["hv"][b]
        dr = s["r"][atoms] @ np.linalg.inv(s["h"][b]) @ dh + dt * out["v"][atoms]
        norm = max(np.linalg.norm(dr, axis=1).max(), np.linalg.norm(dh, axis=1).max())
        scale = min(1.0, maxstep / norm) if norm > 0 else 1.0
        out["r"][atoms] += scale * dr
        out["h"][b] += scale * dh
    return out


def oracle_fire_post(s):
    out = {k: v.copy() for k, v in s.items()}
    for b in range(2):
        atoms = np.where(s["idx"] == b)[0]
        dt = s["dt"][b]
        v = s["v"][atoms] + 0.5 * dt * s["f"][atoms] / s["mass"][atoms, None]
        hv = s["hv"][b] + 0.5 * dt * s["fh"][b]
        vv = np.concatenate((v, hv))
        ff = np.concatenate((s["f"][atoms], s["fh"][b]))
        power = np.sum(vv * ff)
        n = s["n"][b] + 1 if power > 0 else 0
        a = s["a"][b]
        if power > 0 and n >= 2:
            dt = min(dt * 1.1, 0.2)
            a *= 0.99
        elif power <= 0:
            dt = max(dt * 0.5, 0.001)
            a = 0.1
        mix = (
            (1 - a) * vv + a * np.sqrt(np.sum(vv * vv) / np.sum(ff * ff)) * ff
            if np.sum(ff * ff) > 0
            else vv * 0
        )
        out["v"][atoms] = mix[:-3] if power > 0 else 0.0
        out["hv"][b] = mix[-3:] if power > 0 else 0.0
        out["dt"][b] = dt
        out["a"][b] = a
        out["n"][b] = n
    return out


def run_pre(s, maxstep=0.1):
    fire_cell_pre(
        s["r"],
        s["v"],
        s["f"],
        s["mass"],
        s["h"],
        s["hv"],
        s["fh"],
        s["dt"],
        s["idx"],
        maxstep=maxstep,
    )


def run_post(s):
    like = s["a"]
    fire_cell_post(
        s["v"],
        s["f"],
        s["mass"],
        s["hv"],
        s["fh"],
        s["idx"],
        s["a"],
        s["dt"],
        s["n"],
        torch.full_like(like, 0.1),
        torch.full_like(like, 0.99),
        torch.full_like(like, 0.001),
        torch.full_like(like, 0.2),
        torch.full_like(s["n"], 2),
        torch.full_like(like, 0.5),
        torch.full_like(like, 1.1),
    )


@pytest.mark.parametrize("branch", ["positive", "negative", "zero", "cell_only"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_fire_pre_post_numpy_and_all_nine_components(branch, dtype):
    s = state(dtype, branch=branch)
    s["h"][:, 0, 1] = 0.15
    s["h"][:, 0, 2] = -0.12
    s["fh"][:, 0, 2] = 0.2
    before = numpy_state(s)
    expected = oracle_fire_pre(before, 0.0002)
    run_pre(s, 0.0002)
    tol = 2e-12 if dtype == torch.float64 else 3e-5
    for key in ("r", "v", "h", "hv", "dt"):
        np.testing.assert_allclose(s[key].numpy(), expected[key], atol=tol, rtol=tol)
    assert np.linalg.norm(s["h"].numpy() - before["h"], axis=2).max() <= 0.0002 + tol
    expected = oracle_fire_post(numpy_state(s))
    run_post(s)
    for key in ("v", "hv", "a", "dt", "n"):
        np.testing.assert_allclose(s[key].numpy(), expected[key], atol=tol, rtol=tol)


@pytest.mark.parametrize("bad", ["singular", "idx", "grad", "upper", "dt", "alias"])
def test_fire2_invalid_input_before_mutation(bad):
    s = state()
    if bad == "singular":
        s["h"][0].zero_()
    elif bad == "idx":
        s["idx"][0] = 1
    elif bad == "grad":
        s["f"].requires_grad_()
    elif bad == "upper":
        s["h"][0, 0, 1] = 0.2
    elif bad == "dt":
        s["dt"][0] = 0
    else:
        s["hv"] = s["h"]
    before = s["r"].clone()
    with pytest.raises((ValueError, TypeError)):
        run_fire2(s)
    assert torch.equal(s["r"], before)


def test_fire2_preallocated_metadata_external_reductions_and_scale():
    s = state()
    expected = oracle_fire2(numpy_state(s), cell_force_scale=2.0)
    ptr = torch.tensor([0, 2, 5])
    ext_ptr = torch.tensor([0, 4, 9])
    ext_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32)
    ev = torch.empty(9, 3, dtype=s["r"].dtype)
    ef = torch.empty_like(ev)
    scratch = [torch.empty(2, dtype=s["r"].dtype) for _ in range(4)]
    before = {k: v.clone() for k, v in s.items()}
    run_fire2(
        s,
        atom_ptr=ptr,
        ext_atom_ptr=ext_ptr,
        ext_batch_idx=ext_idx,
        ext_velocities=ev,
        ext_forces=ef,
        vf=scratch[0],
        v_sumsq=scratch[1],
        f_sumsq=scratch[2],
        max_norm=scratch[3],
        cell_force_scale=2.0,
    )
    external = [x.clone() for x in scratch[:3]]
    again = {k: v.clone() for k, v in before.items()}
    run_fire2(
        again,
        vf=external[0],
        v_sumsq=external[1],
        f_sumsq=external[2],
        compute_reductions=False,
        cell_force_scale=2.0,
    )
    for key in ("r", "v", "h", "hv", "a", "dt", "n"):
        torch.testing.assert_close(again[key], s[key], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            s[key].numpy(), expected[key], atol=2e-12, rtol=2e-12
        )
    with pytest.raises(ValueError, match="interleaved"):
        run_fire2(state(), ext_batch_idx=torch.tensor([0, 0, 1, 1, 1, 0, 0, 1, 1]))
    with pytest.raises(ValueError, match="required"):
        run_fire2(state(), compute_reductions=False)


def test_fire2_empty_noop_and_reject_empty_member():
    s = state()
    for key in ("r", "v", "f", "idx", "mass"):
        s[key] = s[key][:0]
    before = {k: v.clone() for k, v in s.items()}
    scratch = [torch.ones(2, dtype=s["h"].dtype) for _ in range(4)]
    run_fire2(
        s, vf=scratch[0], v_sumsq=scratch[1], f_sumsq=scratch[2], max_norm=scratch[3]
    )
    assert all(torch.equal(s[k], before[k]) for k in before)
    assert all(not torch.count_nonzero(t) for t in scratch)
    s = state()
    for key in ("r", "v", "f", "idx", "mass"):
        s[key] = s[key][:2].clone()
    with pytest.raises(ValueError, match="at least one atom"):
        run_fire2(s)


def test_full_fire_empty_batch_noop():
    s = state()
    for key in s:
        s[key] = s[key][:0]
    run_pre(s)
    run_post(s)
    assert s["h"].shape == (0, 3, 3)


@pytest.mark.parametrize("runner", [run_fire2, run_post])
def test_control_view_into_velocity_rejected_before_mutation(runner):
    s = state()
    s["v"].fill_(0.1)
    s["a"] = s["v"].reshape(-1)[:2]
    before = {key: value.clone() for key, value in s.items()}
    with pytest.raises(ValueError, match="storage"):
        runner(s)
    assert all(torch.equal(s[key], value) for key, value in before.items())


@pytest.mark.parametrize("runner", [run_fire2, run_pre, run_post])
def test_noncontiguous_state_preserves_identity_and_matches_contiguous(runner):
    original = state()
    strided = {}
    for key, value in original.items():
        storage = torch.empty(*value.shape, 2, dtype=value.dtype)
        strided[key] = storage[..., 0]
        strided[key].copy_(value)
    identities = {key: id(value) for key, value in strided.items()}
    runner(original)
    runner(strided)
    for key in original:
        assert id(strided[key]) == identities[key]
        torch.testing.assert_close(strided[key], original[key], rtol=2e-12, atol=2e-12)
