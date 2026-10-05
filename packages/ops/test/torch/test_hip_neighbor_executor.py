# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-side contract tests for the registered native-HIP neighbor executor."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops._hip_neighbor_executor import neighbor_list
from nvalchemiops.backend import BackendUnavailableError, resolve_backend


def test_hip_periodic_full_matrix_width_is_registered_without_importing_executor():
    selection = resolve_backend(
        "hip",
        operation="neighbor_list",
        device=torch.device("cuda"),
        dtype=torch.float32,
        features={"periodic", "full", "matrix"},
        strategy="cell_list",
    )
    assert selection.implementation_id == "hip.neighbor.cell_list-v1"
    assert selection.family == "hip"
    assert selection.strategy == "cell_list"


def test_hip_executor_rejects_cpu_before_loading_native_extensions():
    positions = torch.zeros((2, 3), dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)
    pbc = torch.ones((1, 3), dtype=torch.bool)
    batch_idx = torch.zeros((2,), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        neighbor_list(
            positions,
            0.5,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            max_neighbors=4,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hip_current_cell_full_rebuild_is_registered(dtype):
    selection = resolve_backend(
        "hip", operation="neighbor_list", device=torch.device("cuda"),
        dtype=dtype, features={"periodic", "full", "matrix", "variable_cell"},
        strategy="cell_list",
    )
    assert selection.implementation_id == "hip.neighbor.cell_list-v1"
    assert selection.gradient_order == 0


def test_hip_stays_explicit_even_when_its_default_strategy_matches_auto():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = resolve_backend(
            "auto", operation="neighbor_list", device=torch.device("cuda"),
            dtype=torch.float64, features={"periodic", "full", "matrix"},
        )
    assert selection.family == "torch_reference"
    # No auto-eligible implementation advertises the new snapshot feature.
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "auto", operation="neighbor_list", device=torch.device("cuda"),
            dtype=torch.float64,
            features={"periodic", "full", "matrix", "variable_cell"},
        )
    # Family and exact implementation selection still admit the default.
    for requested in ("hip", "hip.neighbor.cell_list-v1"):
        explicit = resolve_backend(
            requested, operation="neighbor_list", device=torch.device("cuda"),
            dtype=torch.float64, features={"periodic", "full", "matrix"},
        )
        assert explicit.family == "hip"


@pytest.mark.parametrize("feature", ("no_pbc", "half", "distances", "vectors"))
def test_hip_registry_rejects_unregistered_output_modes(feature: str):
    features = {"periodic", "full", "matrix"}
    if feature == "no_pbc":
        features.remove("periodic")
        features.add("no_pbc")
    else:
        features.add(feature)
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        resolve_backend(
            "hip",
            operation="neighbor_list",
            device=torch.device("cuda"),
            dtype=torch.float32,
            features=features,
            strategy="cell_list",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hip_coo_is_registered_as_matrix_plus_torch_packing(dtype):
    selection = resolve_backend(
        "hip", operation="neighbor_list", device="cuda", dtype=dtype,
        features={"periodic", "full", "coo", "variable_cell"}, strategy="cell_list",
    )
    assert selection.implementation_id == "hip.neighbor.cell_list-v1"
    assert selection.family == "hip"
