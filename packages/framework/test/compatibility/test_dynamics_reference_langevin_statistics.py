"""Short, independent equilibrium check for the fixed-cell Langevin reference."""

from __future__ import annotations

import torch

from nvalchemi._dynamics_reference.langevin import (
    langevin_finalize,
    langevin_half_step,
)


_KB_EV = 8.617333262145e-5


def test_harmonic_batch_samples_target_temperature() -> None:
    """Check position and velocity variances for two harmonic systems.

    The force is the independent analytic oracle ``F = -k*x`` with ``k=1``.
    This intentionally tests moments, not trajectory equality or a MACE model.
    """

    dtype = torch.float64
    atoms_per_system = 128
    num_steps = 2500
    burn_in = 500
    spring_constant = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masses = torch.ones(2 * atoms_per_system, dtype=dtype, device=device)
    batch_idx = torch.cat(
        (
            torch.zeros(atoms_per_system, dtype=torch.int32, device=device),
            torch.ones(atoms_per_system, dtype=torch.int32, device=device),
        )
    )
    temperature = torch.tensor([300.0, 600.0], dtype=dtype, device=device)
    kT = _KB_EV * temperature
    dt = torch.full((2,), 0.02, dtype=dtype, device=device)
    friction = torch.tensor([0.2, 0.5], dtype=dtype, device=device)
    positions = torch.zeros((2 * atoms_per_system, 3), dtype=dtype, device=device)
    velocities = torch.zeros_like(positions)

    position_moments: list[torch.Tensor] = []
    velocity_moments: list[torch.Tensor] = []
    for step in range(num_steps):
        forces = -spring_constant * positions
        langevin_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            kT,
            friction,
            17 + step,
            batch_idx,
        )
        forces_new = -spring_constant * positions
        langevin_finalize(velocities, forces_new, masses, dt, batch_idx)

        if step >= burn_in:
            position_moments.append(
                torch.stack(
                    (
                        positions[:atoms_per_system].square().mean(),
                        positions[atoms_per_system:].square().mean(),
                    )
                )
            )
            velocity_moments.append(
                torch.stack(
                    (
                        velocities[:atoms_per_system].square().mean(),
                        velocities[atoms_per_system:].square().mean(),
                    )
                )
            )

    position_variance = torch.stack(position_moments).mean(dim=0)
    velocity_variance = torch.stack(velocity_moments).mean(dim=0)
    expected_position_variance = kT / spring_constant
    expected_velocity_variance = kT / masses[0]

    # This is a deliberately broad finite-sample tolerance. The test guards
    # the thermostat scale and units without becoming a long benchmark.
    torch.testing.assert_close(
        position_variance,
        expected_position_variance,
        rtol=0.15,
        atol=0.0,
    )
    torch.testing.assert_close(
        velocity_variance,
        expected_velocity_variance,
        rtol=0.15,
        atol=0.0,
    )
