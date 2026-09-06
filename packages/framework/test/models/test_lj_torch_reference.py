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

"""Contract tests for the explicit Warp-independent Lennard-Jones wrapper."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors


def _batch() -> Batch:
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64
                ),
                atomic_numbers=torch.tensor([1, 1]),
            ),
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64
                ),
                atomic_numbers=torch.tensor([1, 1]),
            ),
        ]
    )


def _lj_pair_energy(distance: float) -> float:
    inverse_r = 1.0 / distance
    return 4.0 * (inverse_r**12 - inverse_r**6)


def _lj_force_on_left(distance: float) -> float:
    inverse_r = 1.0 / distance
    return -24.0 / distance * (2.0 * inverse_r**12 - inverse_r**6)


def _evaluate(half_list: bool) -> dict[str, torch.Tensor]:
    batch = _batch()
    compute_neighbors(
        batch,
        cutoff=2.0,
        format=NeighborListFormat.MATRIX,
        half_list=half_list,
        backend="torch_reference",
    )
    model = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
        half_list=half_list,
        backend="torch_reference",
    )
    return model(batch)


def test_lj_reference_import_and_full_half_semantics_without_warp():
    assert "warp" not in sys.modules
    full = _evaluate(half_list=False)
    half = _evaluate(half_list=True)

    expected = torch.tensor(
        [[_lj_pair_energy(1.1)], [_lj_pair_energy(1.2)]], dtype=torch.float64
    )
    expected_forces = torch.tensor(
        [
            [_lj_force_on_left(1.1), 0.0, 0.0],
            [-_lj_force_on_left(1.1), 0.0, 0.0],
            [_lj_force_on_left(1.2), 0.0, 0.0],
            [-_lj_force_on_left(1.2), 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(full["energy"], expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(half["energy"], full["energy"], rtol=1e-12, atol=1e-12)
    assert torch.allclose(full["forces"], half["forces"], rtol=1e-12, atol=1e-12)
    assert torch.allclose(full["forces"], expected_forces, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        full["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12
    )
    assert "warp" not in sys.modules


def test_lj_reference_energy_gradient_matches_returned_force():
    batch = _batch()
    batch.positions.requires_grad_(True)
    compute_neighbors(
        batch,
        cutoff=2.0,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
    )
    model = LennardJonesModelWrapper(1.0, 1.0, 2.0, backend="torch_reference")
    output = model(batch)
    (energy_gradient,) = torch.autograd.grad(output["energy"].sum(), batch.positions)
    assert torch.allclose(
        energy_gradient,
        -output["forces"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_lj_reference_rejects_switch_stress_and_domain_parallel():
    batch = _batch()
    compute_neighbors(
        batch,
        cutoff=2.0,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
    )

    switching = LennardJonesModelWrapper(
        1.0, 1.0, 2.0, switch_width=0.5, backend="torch_reference"
    )
    with pytest.raises(NotImplementedError, match="does not support switching"):
        switching(batch)

    stress = LennardJonesModelWrapper(1.0, 1.0, 2.0, backend="torch_reference")
    stress.model_config.active_outputs.add("stress")
    with pytest.raises(NotImplementedError, match="virial/stress"):
        stress(batch)

    with pytest.raises(NotImplementedError, match="distributed"):
        stress.distribution_spec()


def test_lj_reference_supports_full_periodic_batch_with_shifts():
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
                ),
                atomic_numbers=torch.tensor([1, 1]),
                cell=torch.diag(
                    torch.tensor([3.0, 10.0, 10.0], dtype=torch.float64)
                ).unsqueeze(0),
                pbc=torch.tensor([[True, False, False]]),
            )
        ]
    )
    compute_neighbors(
        batch,
        cutoff=1.5,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
    )
    model = LennardJonesModelWrapper(1.0, 1.0, 1.5, backend="torch_reference")
    output = model(batch)

    expected_energy = torch.tensor([[_lj_pair_energy(1.1)]], dtype=torch.float64)
    expected_forces = torch.tensor(
        [[-_lj_force_on_left(1.1), 0.0, 0.0], [_lj_force_on_left(1.1), 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert model.model_config.supports_pbc is True
    assert torch.allclose(output["energy"], expected_energy, rtol=1e-12, atol=1e-12)
    assert torch.allclose(output["forces"], expected_forces, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        output["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12
    )


def test_reference_model_builds_neighbor_hook_without_eager_dynamics_import():
    model = LennardJonesModelWrapper(1.0, 1.0, 2.0, backend="torch_reference")

    hooks = model.make_neighbor_hooks()

    assert len(hooks) == 1
    assert hooks[0].backend == "torch_reference"
    assert hooks[0].stage.name == "BEFORE_COMPUTE"
    assert "nvalchemi.dynamics" not in sys.modules
    assert "warp" not in sys.modules


def test_reference_neighbor_hook_and_lj_wrapper_form_one_chain():
    batch = _batch()
    model = LennardJonesModelWrapper(1.0, 1.0, 2.0, backend="torch_reference")
    (hook,) = model.make_neighbor_hooks()

    hook(HookContext(batch=batch), hook.stage)
    output = model(batch)

    expected = torch.tensor(
        [[_lj_pair_energy(1.1)], [_lj_pair_energy(1.2)]], dtype=torch.float64
    )
    assert torch.allclose(output["energy"], expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        output["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12
    )
