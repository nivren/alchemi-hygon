# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata for fixed-cell Torch-reference dynamics operations."""

from nvalchemiops._backend_types import Implementation

IMPLEMENTATIONS: tuple[Implementation, ...] = (
    Implementation(
        implementation_id="torch_reference.velocity_verlet-v1",
        operation="velocity_verlet",
        family="torch_reference",
        executor="nvalchemi._dynamics_reference.velocity_verlet",
        entrypoints=("vv_position_update", "vv_velocity_finalize"),
        executor_owner="framework",
        features=frozenset({"fixed_cell"}),
        max_gradient_order=1,
        evidence="G2 velocity-Verlet reference contracts",
        default_strategy=True,
    ),
    Implementation(
        implementation_id="torch_reference.fire-v1",
        operation="fire",
        family="torch_reference",
        executor="nvalchemi._dynamics_reference.fire",
        entrypoints=("fire_step", "fire_update"),
        executor_owner="framework",
        features=frozenset({"fixed_cell"}),
        max_gradient_order=1,
        evidence="G2 FIRE/FIRE2 reference contracts",
        default_strategy=True,
    ),
    Implementation(
        implementation_id="torch_reference.fire2-v1",
        operation="fire2",
        family="torch_reference",
        executor="nvalchemi._dynamics_reference.fire",
        entrypoints=("fire2_step_coord",),
        executor_owner="framework",
        features=frozenset({"fixed_cell"}),
        max_gradient_order=1,
        evidence="G2 fixed-cell FIRE2 reference contracts",
        default_strategy=True,
    ),
    Implementation(
        implementation_id="torch_reference.kinetics-v1",
        operation="kinetics",
        family="torch_reference",
        executor="nvalchemi._dynamics_reference.kinetics",
        entrypoints=("kinetic_energy_per_graph", "temperature_per_graph"),
        executor_owner="framework",
        features=frozenset({"per_graph"}),
        max_gradient_order=1,
        evidence="G2 kinetic-energy/temperature reference contracts",
        default_strategy=True,
    ),
)
