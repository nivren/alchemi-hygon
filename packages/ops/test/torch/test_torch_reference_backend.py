# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for the Warp-independent Torch reference backend."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_import_does_not_load_warp():
    """Torch reference operators must be usable without importing Warp."""
    package_root = Path(__file__).resolve().parents[2]
    script = (
        "import sys; import nvalchemiops; "
        "assert 'warp' not in sys.modules; "
        "from nvalchemiops.torch_reference import neighbor_list; "
        "assert callable(neighbor_list); assert 'warp' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_root)
    subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_neighbor_list_batch_full_half_and_coo():
    """Full and half lists preserve batches and expose consistent COO data."""
    from nvalchemiops.torch_reference import neighbor_list  # noqa: PLC0415

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 2.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)

    full, full_count = neighbor_list(positions, 2.0, batch_ptr=batch_ptr)
    half, half_count = neighbor_list(
        positions, 2.0, batch_ptr=batch_ptr, half_fill=True
    )
    assert full.tolist() == [[1], [0], [4], [4]]
    assert full_count.tolist() == [1, 1, 0, 0]
    assert half.tolist() == [[1], [4], [4], [4]]
    assert half_count.tolist() == [1, 0, 0, 0]

    coo, ptr, distances, vectors = neighbor_list(
        positions,
        2.0,
        batch_ptr=batch_ptr,
        return_neighbor_list=True,
        return_distances=True,
        return_vectors=True,
    )
    assert coo.tolist() == [[0, 1], [1, 0]]
    assert ptr.tolist() == [0, 1, 2, 2, 2]
    assert torch.allclose(distances, torch.tensor([1.1, 1.1], dtype=torch.float64))
    assert torch.allclose(
        vectors,
        torch.tensor([[-1.1, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64),
    )


def test_lj_energy_force_and_second_derivative_match_list_conventions():
    """LJ energy and force agree for full/half lists and retain grad history."""
    from nvalchemiops.torch_reference import (  # noqa: PLC0415
        lj_energy_forces,
        neighbor_list,
    )

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 2.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    results = []
    for half in (False, True):
        matrix, counts = neighbor_list(positions, 2.0, half_fill=half)
        coordinates = positions.clone().requires_grad_()
        atomic, forces = lj_energy_forces(
            coordinates,
            matrix,
            counts,
            epsilon=1.0,
            sigma=1.0,
            cutoff=2.0,
            half_list=half,
        )
        curvature = torch.autograd.grad(
            (forces.square()).sum(), coordinates, retain_graph=False
        )[0]
        assert curvature.shape == coordinates.shape
        results.append((atomic, forces))

    atomic_full, forces_full = results[0]
    atomic_half, forces_half = results[1]
    assert torch.allclose(atomic_full, atomic_half, atol=1e-12, rtol=1e-12)
    assert torch.allclose(forces_full, forces_half, atol=1e-12, rtol=1e-12)
    assert torch.allclose(atomic_full.sum(), torch.tensor(-0.9833724493736827, dtype=torch.float64))


def test_neighbor_capacity_and_unsupported_features_are_explicit():
    """The reference backend never truncates neighbors or silently ignores PBC."""
    from nvalchemiops.torch_reference import (  # noqa: PLC0415
        NeighborOverflowError,
        neighbor_list,
    )

    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(NeighborOverflowError):
        neighbor_list(positions, 2.0, max_neighbors=0)
    with pytest.raises(NotImplementedError, match="no PBC"):
        neighbor_list(positions, 2.0, pbc=torch.ones(3, dtype=torch.bool))


def test_single_atom_zero_capacity_is_valid():
    """An isolated one-atom system has a valid zero-width neighbor matrix."""
    from nvalchemiops.torch_reference import neighbor_list  # noqa: PLC0415

    matrix, counts = neighbor_list(torch.zeros((1, 3)), 2.0)
    assert matrix.shape == (1, 0)
    assert counts.tolist() == [0]
