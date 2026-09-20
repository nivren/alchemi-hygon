# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata for Torch-reference neighbor-list strategies."""

from nvalchemiops._backend_types import Implementation

IMPLEMENTATIONS: tuple[Implementation, ...] = (
    Implementation(
        implementation_id="torch_reference.neighbor.dense-v1",
        operation="neighbor_list",
        family="torch_reference",
        strategy="dense",
        executor="nvalchemiops.torch_reference",
        entrypoints=("neighbor_list",),
        executor_owner="ops",
        features=frozenset(
            {
                "no_pbc",
                "periodic",
                "full",
                "half",
                "matrix",
                "coo",
                "distances",
                "vectors",
            }
        ),
        excluded_feature_sets=(frozenset({"periodic", "half"}),),
        evidence="G1/G2 Torch-reference neighbor contracts",
        default_strategy=True,
    ),
    Implementation(
        implementation_id="torch_reference.neighbor.cell_list-v1",
        operation="neighbor_list",
        family="torch_reference",
        strategy="cell_list",
        executor="nvalchemiops.torch_reference_cell_list",
        entrypoints=("neighbor_list",),
        executor_owner="ops",
        features=frozenset(
            {
                "no_pbc",
                "periodic",
                "full",
                "half",
                "matrix",
                "coo",
                "distances",
                "vectors",
            }
        ),
        max_gradient_order=2,
        evidence="Torch cell-list core contract: no-PBC/PBC full/half build-query",
    ),
    Implementation(
        implementation_id="hip.neighbor.cell_list-v1",
        operation="neighbor_list",
        family="hip",
        strategy="cell_list",
        executor="nvalchemiops._hip_neighbor_executor",
        entrypoints=("neighbor_list",),
        executor_owner="ops",
        features=frozenset({"periodic", "full", "matrix"}),
        dtypes=frozenset({"float32", "float64"}),
        max_gradient_order=0,
        devices=frozenset({"cuda"}),
        evidence="Point38 fixed-cell/full-list native HIP framework executor boundary",
        default_strategy=True,
    ),
)
