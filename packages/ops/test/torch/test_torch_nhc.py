"""Independent NHC formulas, functional gradients and upstream CPU parity."""

import math

import pytest
import torch

from nvalchemiops import torch_nhc as nhc


def state(dtype=torch.float64, C=3):
    v = torch.arange(18, dtype=dtype).reshape(6, 3) / 50
    mass = torch.arange(1, 7, dtype=dtype)
    idx = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32)
    kT = torch.tensor([0.03, 0.06], dtype=dtype)
    dt = torch.tensor([0.02, 0.03], dtype=dtype)
    tau = torch.tensor([0.8, 1.1], dtype=dtype)
    Q = nhc.nhc_compute_masses(kT, tau, mass, idx, C)
    eta = torch.zeros_like(Q)
    u = torch.ones_like(Q) * 0.01
    ndof = torch.tensor([6.0, 12.0], dtype=dtype)
    return v, mass, eta, u, Q, kT, dt, ndof, idx


def scalar_oracle(v, mass, eta, u, Q, kT, dt, ndof):
    """One-link closed scalar step; independent of tensor chain implementation."""
    kinetic = sum(m * sum(x * x for x in row) for m, row in zip(mass, v))
    e = eta + 0.25 * dt * u
    u += 0.25 * dt * (kinetic - ndof * kT) / Q
    scale = math.exp(-0.5 * dt * u)
    kinetic *= scale * scale
    u += 0.25 * dt * (kinetic - ndof * kT) / Q
    e += 0.25 * dt * u
    return [[x * scale for x in row] for row in v], e, u


def test_mass_and_kick_drift_formulas():
    v, mass, eta, u, Q, kT, dt, ndof, idx = state()
    expected = kT * torch.tensor([0.8, 1.1], dtype=v.dtype).square()
    torch.testing.assert_close(Q[:, 0], ndof * expected)
    torch.testing.assert_close(Q[:, 1], expected)
    forces = torch.ones_like(v)
    actual = nhc.nhc_velocity_half_step_functional(v, forces, mass, dt, idx)
    torch.testing.assert_close(actual, v + 0.5 * dt[idx, None] / mass[:, None])
    torch.testing.assert_close(
        nhc.nhc_position_update_functional(v, actual, dt, idx),
        v + dt[idx, None] * actual,
    )


def test_single_link_matches_independent_scalar_formula():
    args = state(C=1)
    actual = nhc.nhc_chain_update_functional(*args)
    v, mass, eta, u, Q, kT, dt, ndof, idx = args
    for s in range(2):
        mask = idx == s
        vv, e, udot = scalar_oracle(
            v[mask].tolist(),
            mass[mask].tolist(),
            float(eta[s, 0]),
            float(u[s, 0]),
            float(Q[s, 0]),
            float(kT[s]),
            float(dt[s]),
            float(ndof[s]),
        )
        torch.testing.assert_close(actual[0][mask], torch.tensor(vv, dtype=v.dtype))
        assert float(actual[1][s, 0]) == pytest.approx(e)
        assert float(actual[2][s, 0]) == pytest.approx(udot)


@pytest.mark.parametrize("C", [1, 3, 8])
@pytest.mark.parametrize("nloops", [1, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chain_reversible_and_external_ke(C, nloops, dtype):
    args = state(dtype, C)
    forward = nhc.nhc_chain_update_functional(*args, nloops=nloops)
    v, mass, eta, u, Q, kT, dt, ndof, idx = args
    external = nhc.nhc_chain_update_functional(
        *args, nloops=nloops, ke2=nhc.nhc_compute_2ke(v, mass, idx, 2)
    )
    for a, b in zip(forward, external):
        torch.testing.assert_close(a, b)
    backward = nhc.nhc_chain_update_functional(
        forward[0], mass, forward[1], forward[2], Q, kT, -dt, ndof, idx, nloops=nloops
    )
    for original, recovered in zip((v, eta, u), backward[:3]):
        torch.testing.assert_close(
            original,
            recovered,
            rtol=2e-5 if dtype == torch.float32 else 1e-10,
            atol=2e-7 if dtype == torch.float32 else 1e-12,
        )


def test_functional_first_and_second_gradients():
    args = list(state(C=2))
    v = args[0].requires_grad_()

    def function(value):
        current = list(args)
        current[0] = value
        out = nhc.nhc_chain_update_functional(*current, nloops=3)
        return out[0], out[1], out[2]

    assert torch.autograd.gradcheck(function, (v,))
    assert torch.autograd.gradgradcheck(function, (v,))


@pytest.mark.parametrize("nloops", [1, 3, 5])
def test_corrected_warp_cpu_parity(nloops):
    wp = pytest.importorskip("warp")
    from nvalchemiops.dynamics.integrators.nose_hoover import (
        nhc_thermostat_chain_update,
    )

    args = state()
    expected = nhc.nhc_chain_update_functional(*args, nloops=nloops)
    v, mass, eta, u, Q, kT, dt, ndof, idx = [x.clone() for x in args]
    scratch = [torch.zeros(2, dtype=v.dtype) for _ in range(4)]
    scratch[1].fill_(1.0)
    nhc_thermostat_chain_update(
        wp.from_torch(v, dtype=wp.vec3d),
        wp.from_torch(mass, dtype=wp.float64),
        wp.from_torch(eta, dtype=wp.float64),
        wp.from_torch(u, dtype=wp.float64),
        wp.from_torch(Q, dtype=wp.float64),
        wp.from_torch(kT, dtype=wp.float64),
        wp.from_torch(dt, dtype=wp.float64),
        wp.from_torch(ndof, dtype=wp.float64),
        *[wp.from_torch(x, dtype=wp.float64) for x in scratch],
        nloops=nloops,
        batch_idx=wp.from_torch(idx, dtype=wp.int32),
        num_systems=2,
        device="cpu",
    )
    for a, b in zip((v, eta, u, *scratch), expected):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-13)


def test_empty_and_invalid_contract():
    args = list(state())
    args[4][0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        nhc.nhc_chain_update_functional(*args)
    args = list(state())
    args[-1][0] = 2
    with pytest.raises(ValueError, match="out-of-range"):
        nhc.nhc_chain_update_functional(*args)
    for C in (0, 9):
        with pytest.raises(ValueError, match="chain_length"):
            nhc.nhc_compute_masses(args[5], args[6], args[1], args[-1], C)
    empty = [
        torch.empty(0, 3, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, 3, dtype=torch.float64),
        torch.empty(0, 3, dtype=torch.float64),
        torch.empty(0, 3, dtype=torch.float64),
        *[torch.empty(0, dtype=torch.float64) for _ in range(3)],
        torch.empty(0, dtype=torch.int32),
    ]
    assert nhc.nhc_chain_update_functional(*empty)[0].shape == (0, 3)
