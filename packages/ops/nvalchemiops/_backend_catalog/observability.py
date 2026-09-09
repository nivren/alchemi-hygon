# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata for periodic and observer helper operations."""

from nvalchemiops._backend_types import Implementation

IMPLEMENTATIONS: tuple[Implementation, ...] = (
    Implementation(
        implementation_id="torch_reference.periodic_wrap-v1",
        operation="periodic_wrap",
        family="torch_reference",
        executor="nvalchemi._dynamics_reference.periodic",
        entrypoints=("wrap_positions_into_cell",),
        executor_owner="framework",
        features=frozenset({"inplace", "periodic"}),
        evidence="G2 periodic-hook reference contracts",
        default_strategy=True,
    ),
    Implementation(
        implementation_id="torch_reference.segmented_reduce-v1",
        operation="segmented_reduce",
        family="torch_reference",
        executor="nvalchemi._dynamics_reference.segmented_reduce",
        entrypoints=("scatter_reduce_per_graph",),
        executor_owner="framework",
        features=frozenset({"per_graph"}),
        evidence="G2 observer reference contracts",
        default_strategy=True,
    ),
)
