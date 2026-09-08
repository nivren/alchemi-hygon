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


def test_no_pbc_vectorized_assembly_preserves_row_order_and_error_contract():
    """Tier-1 no-PBC assembly keeps the former matrix/COO semantics exactly."""
    from nvalchemiops.torch_reference import NeighborOverflowError, neighbor_list

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    matrix, counts, distances, vectors = neighbor_list(
        positions,
        1.5,
        batch_ptr=torch.tensor([0, 3, 4]),
        return_distances=True,
        return_vectors=True,
    )
    assert matrix.tolist() == [[1, 4], [0, 2], [1, 4], [4, 4]]
    assert counts.tolist() == [1, 2, 1, 0]
    assert distances.tolist() == [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    assert vectors[:, 0].tolist() == [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    with pytest.raises(NeighborOverflowError, match="row 1 count 2"):
        neighbor_list(positions, 1.5, batch_ptr=torch.tensor([0, 3, 4]), max_neighbors=1)
    with pytest.raises(ValueError, match="overlapping active pair"):
        neighbor_list(torch.zeros(2, 3), 1.0)


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
    """The reference backend never truncates neighbors or silently ignores limits."""
    from nvalchemiops.torch_reference import (  # noqa: PLC0415
        NeighborOverflowError,
        neighbor_list,
    )

    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(NeighborOverflowError):
        neighbor_list(positions, 2.0, max_neighbors=0)
    with pytest.raises(ValueError, match="cell and pbc"):
        neighbor_list(positions, 2.0, pbc=torch.ones(3, dtype=torch.bool))
    with pytest.raises(NotImplementedError, match="half_fill=False"):
        neighbor_list(
            positions,
            2.0,
            cell=torch.eye(3),
            pbc=torch.ones(3, dtype=torch.bool),
            half_fill=True,
        )


def test_periodic_full_list_returns_integer_image_shifts():
    """Periodic full lists include the nearest image and its signed shift."""
    from nvalchemiops.torch_reference import neighbor_list  # noqa: PLC0415

    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64
    )
    matrix, counts, shifts = neighbor_list(
        positions,
        0.5,
        cell=torch.diag(torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)),
        pbc=torch.tensor([True, False, False]),
    )
    assert matrix.tolist() == [[1], [0]]
    assert counts.tolist() == [1, 1]
    assert shifts.tolist() == [[[-1, 0, 0]], [[1, 0, 0]]]

    edges, ptr, coo_shifts = neighbor_list(
        positions,
        0.5,
        cell=torch.diag(torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)),
        pbc=torch.tensor([True, False, False]),
        return_neighbor_list=True,
    )
    assert edges.tolist() == [[0, 1], [1, 0]]
    assert ptr.tolist() == [0, 1, 2]
    assert coo_shifts.tolist() == [[-1, 0, 0], [1, 0, 0]]


def test_periodic_triclinic_cell_preserves_shift_contract():
    """The reference shift calculation also handles a non-orthogonal cell."""
    from nvalchemiops.torch_reference import neighbor_list  # noqa: PLC0415

    positions = torch.tensor(
        [[0.1, 0.1, 0.0], [0.55, 1.9, 0.0]], dtype=torch.float64
    )
    cell = torch.tensor(
        [[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=torch.float64,
    )
    matrix, counts, shifts = neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool),
    )
    assert matrix.tolist() == [[1], [0]]
    assert counts.tolist() == [1, 1]
    assert shifts.tolist() == [[[0, -1, 0]], [[0, 1, 0]]]


def test_periodic_neighbor_contract_covers_images_batches_and_boundaries():
    """Periodic topology keeps all valid images and rejects invalid rows."""
    from nvalchemiops.torch_reference import (  # noqa: PLC0415
        NeighborOverflowError,
        neighbor_list,
    )

    single_atom = torch.zeros((1, 3), dtype=torch.float64)
    matrix, counts, shifts = neighbor_list(
        single_atom,
        1.1,
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, False, False]),
    )
    assert matrix.tolist() == [[0, 0]]
    assert counts.tolist() == [2]
    assert {tuple(shift) for shift in shifts[0].tolist()} == {(-1, 0, 0), (1, 0, 0)}

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    matrix, counts, shifts = neighbor_list(
        positions,
        1.0,
        cell=torch.stack((torch.eye(3), torch.eye(3))).to(torch.float64) * 10.0,
        pbc=torch.tensor([[True, False, False], [False, False, False]]),
        batch_ptr=torch.tensor([0, 2, 3]),
    )
    assert counts.tolist() == [0, 0, 0]
    assert matrix.shape == (3, 0)
    assert shifts.shape == (3, 0, 3)

    with pytest.raises(NeighborOverflowError):
        neighbor_list(
            single_atom,
            1.1,
            cell=torch.eye(3),
            pbc=torch.tensor([True, False, False]),
            max_neighbors=1,
        )
    with pytest.raises(ValueError, match="overlapping active pair"):
        neighbor_list(
            torch.zeros((2, 3)),
            1.0,
            cell=torch.eye(3),
            pbc=torch.tensor([True, False, False]),
        )


def test_periodic_lj_matches_independent_fp64_pair_oracle():
    """Periodic LJ consumes signed shifts without changing force semantics."""
    from nvalchemiops.torch_reference import (  # noqa: PLC0415
        lj_energy_forces,
        neighbor_list,
    )

    epsilon = 1.0
    sigma = 1.0
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
    )
    cell = torch.diag(torch.tensor([3.0, 10.0, 10.0], dtype=torch.float64))
    matrix, counts, shifts = neighbor_list(
        positions,
        1.5,
        cell=cell,
        pbc=torch.tensor([True, False, False]),
    )
    coordinates = positions.clone().requires_grad_()
    atomic, forces = lj_energy_forces(
        coordinates,
        matrix,
        counts,
        epsilon=epsilon,
        sigma=sigma,
        cutoff=1.5,
        cell=cell,
        batch_idx=torch.zeros(2, dtype=torch.int32),
        neighbor_matrix_shifts=shifts,
    )

    distance = torch.tensor(1.1, dtype=torch.float64)
    inverse_r = 1.0 / distance
    expected_energy = 4.0 * (inverse_r.pow(12) - inverse_r.pow(6))
    expected_force_x = 24.0 / distance * (2.0 * inverse_r.pow(12) - inverse_r.pow(6))
    expected_forces = torch.tensor(
        [[expected_force_x, 0.0, 0.0], [-expected_force_x, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(atomic.sum(), expected_energy, rtol=1e-12, atol=1e-12)
    assert torch.allclose(forces, expected_forces, rtol=1e-12, atol=1e-12)
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    (gradient,) = torch.autograd.grad(atomic.sum(), coordinates)
    assert torch.allclose(gradient, -forces, rtol=1e-12, atol=1e-12)


def test_single_atom_zero_capacity_is_valid():
    """An isolated one-atom system has a valid zero-width neighbor matrix."""
    from nvalchemiops.torch_reference import neighbor_list  # noqa: PLC0415

    matrix, counts = neighbor_list(torch.zeros((1, 3)), 2.0)
    assert matrix.shape == (1, 0)
    assert counts.tolist() == [0]


def test_dispatcher_preserves_outputs_and_reports_backend():
    """The explicit dispatcher keeps tensor outputs unchanged and auditable."""
    from nvalchemiops.backend import (  # noqa: PLC0415
        BackendUnavailableError,
        resolve_backend,
    )
    from nvalchemiops.torch_backend import (  # noqa: PLC0415
        dispatch_lj_energy_forces,
        dispatch_neighbor_list,
    )

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=torch.float64,
    )
    (matrix, counts), neighbor_selection = dispatch_neighbor_list(
        positions,
        2.0,
        backend="auto",
        return_backend=True,
    )
    assert neighbor_selection.as_dict() == {
        "requested": "auto",
        "selected": "torch_reference",
        "operation": "neighbor_list",
        "device": "cpu",
        "dtype": "float64",
        "gradient_order": 0,
        "features": ["full", "matrix", "no_pbc"],
        "reason": "auto selected the highest-priority verified capability",
    }

    coordinates = positions.clone().requires_grad_()
    (atomic, forces), lj_selection = dispatch_lj_energy_forces(
        coordinates,
        matrix,
        counts,
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
        backend="torch_reference",
        return_backend=True,
    )
    assert atomic.shape == (2,)
    assert forces.shape == (2, 3)
    assert lj_selection.selected == "torch_reference"
    assert lj_selection.operation == "lj_energy_forces"
    assert resolve_backend("torch_reference", operation="neighbor_list").device == "unspecified"
    with pytest.raises(BackendUnavailableError, match="no verified capability"):
        dispatch_neighbor_list(positions, 2.0, backend="triton")
