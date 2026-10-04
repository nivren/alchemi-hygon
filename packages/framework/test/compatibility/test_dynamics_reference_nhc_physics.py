"""NHC physical oracles independent of learned potentials."""

import torch
from nvalchemiops.torch_nhc import (
    nhc_chain_update_functional as chain,
)
from nvalchemiops.torch_nhc import (
    nhc_compute_chain_energy,
    nhc_compute_masses,
)


def simulate(dt, steps):
    generator = torch.Generator().manual_seed(53)
    n = 128
    idx = torch.arange(2).repeat_interleave(n)
    dtype = torch.float64
    mass = torch.ones(2 * n, dtype=dtype)
    kT = torch.tensor([300.0, 600.0], dtype=dtype) * 8.617333262145e-5
    spring = torch.linspace(0.6, 1.4, 2 * n, dtype=dtype)[:, None].square()
    v = (
        torch.randn(2 * n, 3, generator=generator, dtype=dtype)
        * torch.sqrt(kT[idx, None])
        * 0.7
    )
    p = (
        torch.randn(2 * n, 3, generator=generator, dtype=dtype)
        * torch.sqrt(kT[idx, None] / spring)
        * 0.7
    )
    Q = nhc_compute_masses(kT, torch.tensor([0.5, 0.7], dtype=dtype), mass, idx, 3)
    eta = torch.zeros_like(Q)
    u = torch.zeros_like(Q)
    ndof = kT.new_full((2,), 3 * n)
    dts = kT.new_full((2,), dt)
    moments = []
    energies = []
    for step in range(steps):
        v, eta, u, *_ = chain(v, mass, eta, u, Q, kT, dts, ndof, idx, nloops=3)
        v = v - 0.5 * dts[idx, None] * spring * p
        p = p + dts[idx, None] * v
        v = v - 0.5 * dts[idx, None] * spring * p
        v, eta, u, *_ = chain(v, mass, eta, u, Q, kT, dts, ndof, idx, nloops=3)
        kinetic, potential = nhc_compute_chain_energy(eta, u, Q, kT, ndof)
        atom = 0.5 * (v.square() + spring * p.square()).sum(1)
        energies.append(atom.new_zeros(2).index_add(0, idx, atom) + kinetic + potential)
        if step >= steps // 4:
            moments.append(
                torch.stack(
                    [
                        v.square().reshape(2, n, 3).mean((1, 2)),
                        (spring * p.square()).reshape(2, n, 3).mean((1, 2)),
                    ]
                )
            )
    energy = torch.stack(energies)
    return torch.stack(moments).mean(0) / kT, energy.max(0).values - energy.min(
        0
    ).values


def test_thermostat_half_step_eta_derivative():
    # Continuous NHC equation: d eta / dt = eta_dot. A half operator of
    # duration dt/2 must advance eta by eta_dot*dt/2 to first order.
    dtype = torch.float64
    dt = torch.tensor([1e-7], dtype=dtype)
    v = torch.ones(2, 3, dtype=dtype) * 0.1
    mass = torch.ones(2, dtype=dtype)
    idx = torch.zeros(2, dtype=torch.long)
    kT = torch.tensor([0.03], dtype=dtype)
    Q = nhc_compute_masses(kT, torch.ones_like(kT), mass, idx, 3)
    eta = torch.zeros_like(Q)
    u = torch.tensor([[0.1, 0.2, 0.3]], dtype=dtype)
    out = chain(v, mass, eta, u, Q, kT, dt, torch.tensor([6.0], dtype=dtype), idx)
    torch.testing.assert_close(out[1] / dt[:, None], u * 0.5, rtol=1e-6, atol=1e-8)


def test_harmonic_statistics_and_extended_energy_step_convergence():
    moments, coarse = simulate(0.02, 2000)
    fine_moments, fine = simulate(0.01, 4000)
    torch.testing.assert_close(moments, torch.ones_like(moments), rtol=0.05, atol=0.0)
    torch.testing.assert_close(
        fine_moments, torch.ones_like(moments), rtol=0.05, atol=0.0
    )
    assert (coarse < 0.001).all()
    # Same physical duration, dt/2: expected second-order bounded energy error.
    assert ((fine / coarse > 0.20) & (fine / coarse < 0.30)).all()
    print(
        {
            "moments_ratio": moments.tolist(),
            "coarse_energy_range": coarse.tolist(),
            "fine_energy_range": fine.tolist(),
        }
    )
