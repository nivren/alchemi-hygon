# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata for Torch-reference interaction operations."""

from nvalchemiops._backend_types import Implementation

IMPLEMENTATIONS: tuple[Implementation, ...] = (
    Implementation(
        implementation_id="torch_reference.lj_energy_forces-v1",
        operation="lj_energy_forces",
        family="torch_reference",
        executor="nvalchemiops.torch_reference.lj_energy_forces",
        features=frozenset({"no_pbc", "periodic", "full", "half", "forces"}),
        excluded_feature_sets=(frozenset({"periodic", "half"}),),
        max_gradient_order=2,
        evidence="G1 Torch-reference LJ force/curvature contracts",
        default_strategy=True,
    ),
)
