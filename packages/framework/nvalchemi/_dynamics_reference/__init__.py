# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-free reference implementations for dynamics primitives.

This package intentionally lives at the top level of :mod:`nvalchemi`.  It can
therefore be imported without executing ``nvalchemi.dynamics`` and its legacy
Warp-backed import graph.  Public dynamics classes will delegate here only
after their backend boundary has been isolated.
"""

from nvalchemi._dynamics_reference.velocity_verlet import (
    vv_position_update,
    vv_velocity_finalize,
)
from nvalchemi._dynamics_reference.langevin import (
    langevin_finalize,
    langevin_half_step,
)
from nvalchemi._dynamics_reference.kinetics import (
    KB_EV,
    kinetic_energy_per_graph,
    temperature_per_graph,
)
from nvalchemi._dynamics_reference.fire import (
    fire2_step_coord,
    fire2_step_coord_cell,
    fire_step,
    fire_update,
)
from nvalchemi._dynamics_reference.periodic import wrap_positions_into_cell

__all__ = [
    "KB_EV",
    "kinetic_energy_per_graph",
    "temperature_per_graph",
    "fire_step",
    "fire_update",
    "fire2_step_coord",
    "fire2_step_coord_cell",
    "vv_position_update",
    "vv_velocity_finalize",
    "langevin_half_step",
    "langevin_finalize",
    "wrap_positions_into_cell",
]
