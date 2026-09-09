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
        executor="nvalchemiops.torch_reference.neighbor_list",
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
        executor="nvalchemiops.torch_reference_cell_list.neighbor_list",
        features=frozenset(
            {"no_pbc", "full", "half", "matrix", "coo", "distances", "vectors"}
        ),
        evidence="G2 opt-in no-PBC Torch reference cell-list contract",
    ),
)
