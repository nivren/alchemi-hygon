# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
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

"""ASE 3.29 step oracle and local BFGS state/backend failure contracts."""

import dataclasses
import io
import os
import subprocess
import sys
import types

import ase
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS

from nvalchemiops import backend as registry_api
from nvalchemiops.backend import (
    BackendUnavailableError,
    Implementation,
    resolve_backend,
)
from nvalchemiops.bfgs_ase import (
    ASEBFGSState,
    BFGSNumericalError,
    bfgs_ase_prepare_state,
    bfgs_ase_step,
)
from nvalchemiops.executor import clear_entrypoint_cache


def assert_snapshot_equal(state, snapshot):
    """Check failure atomicity, including all historical tensor values."""
    actual = state.state_dict()
    assert actual.keys() == snapshot.keys()
    for name, value in snapshot.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        else:
            assert actual[name] == value


def potential_force(q, kind):
    """Independent smooth test potentials with coupled positive curvature."""
    flat = q.ravel()
    diagonal = np.linspace(4.0, 14.0, flat.size)
    matrix = np.diag(diagonal) + 0.7 * np.ones((flat.size, flat.size))
    quartic = 0.6 * flat**3 if kind == "quartic" else 0.0
    return (-matrix @ flat - quartic).reshape(q.shape)


@pytest.mark.parametrize(
    "kind,alpha,maxstep",
    [("quadratic", 70.0, 0.2), ("quartic", 22.0, 0.09), ("quadratic", 35.0, 2.0)],
)
def test_ase_multistep_oracle(kind, alpha, maxstep):
    """Compare evolving independent trajectories and every internal quantity."""
    assert ase.__version__ == "3.29.0", (
        "Revalidate the frozen ASE contract before upgrading"
    )
    initial = np.array([[1.3, -0.7, 0.8], [-0.6, 1.1, -0.9], [0.2, -0.4, 0.5]])
    atoms = Atoms("H3", positions=initial)
    oracle = BFGS(atoms, logfile=None, alpha=alpha, maxstep=maxstep)
    q = torch.tensor(initial, dtype=torch.float64)
    state = bfgs_ase_prepare_state(
        q.numel(), alpha=alpha, maxstep=maxstep, system_id=17
    )
    for _ in range(8):
        before = atoms.get_positions()
        force = potential_force(before, kind)
        raw, lengths = oracle.prepare_step(before.ravel(), -force.ravel())
        raw = raw.copy()
        applied = oracle.determine_step(raw.copy(), lengths)
        atoms.set_positions(before + applied.reshape(-1, 3))
        actual_force = potential_force(q.numpy(), kind)
        result = bfgs_ase_step(q, torch.tensor(actual_force), state, system_id=17)
        expected = {
            "hessian": (state.hessian, oracle.H),
            "spectrum": (result.eigenvalues, np.linalg.eigvalsh(oracle.H)),
            "raw_step": (result.raw_step, raw.reshape(-1, 3)),
            "applied_step": (result.applied_step, applied.reshape(-1, 3)),
            "coordinates": (q, atoms.positions),
            "history_q": (state.previous_positions, oracle.pos0.reshape(-1, 3)),
            "history_f": (state.previous_forces, oracle.forces0.reshape(-1, 3)),
        }
        for name, (actual, target) in expected.items():
            torch.testing.assert_close(
                actual, torch.tensor(target.copy()), rtol=1e-10, atol=1e-12, msg=name
            )
        assert result.scale.item() == pytest.approx(
            min(1.0, maxstep / np.max(lengths)), rel=1e-10
        )
    assert state.step_count == 8 and state.initialized


def test_ase_public_step_updates_coordinates_and_history():
    """Exercise ASE's public step, beyond its numerical prepare/determine methods."""
    initial = np.array([[1.2, -0.6, 0.7], [-0.4, 0.8, -0.9]])
    atoms = Atoms("H2", positions=initial)

    class QuarticCalculator(Calculator):
        """Analytic ASE energy/force source, without deprecated step arguments."""

        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            """Evaluate the coupled potential independently of BFGS."""
            super().calculate(atoms, properties, system_changes)
            x = self.atoms.positions.ravel()
            matrix = np.diag(np.linspace(4.0, 14.0, x.size)) + 0.7 * np.ones(
                (x.size, x.size)
            )
            self.results = {
                "energy": 0.5 * x @ matrix @ x + 0.15 * np.sum(x**4),
                "forces": potential_force(self.atoms.positions, "quartic"),
            }

    atoms.calc = QuarticCalculator()
    oracle = BFGS(atoms, logfile=None)
    q = torch.tensor(initial)
    state = bfgs_ase_prepare_state(6)
    for _ in range(8):
        oracle.step()
        bfgs_ase_step(q, torch.tensor(potential_force(q.numpy(), "quartic")), state)
        torch.testing.assert_close(
            q, torch.tensor(atoms.positions), rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(
            state.hessian, torch.tensor(oracle.H), rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(
            state.previous_positions,
            torch.tensor(oracle.pos0.reshape(-1, 3)),
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.mark.parametrize("force", [0.0, 0.3, 0.6, 1.2])
def test_global_maxstep_and_zero_force(force):
    q = torch.zeros((2, 3), dtype=torch.float64)
    f = torch.tensor([[force, 0.0, 0.0], [0.0, force / 2.0, 0.0]], dtype=q.dtype)
    original_force = f.clone()
    result = bfgs_ase_step(q, f, bfgs_ase_prepare_state(6, alpha=3.0, maxstep=0.2))
    scale = 1.0 if force <= 0.6 else 0.6 / force
    torch.testing.assert_close(q, f / 3.0 * scale)
    torch.testing.assert_close(result.raw_step, f / 3.0)
    assert result.scale.item() == pytest.approx(scale)
    torch.testing.assert_close(f, original_force, rtol=0, atol=0)
    result.applied_step.add_(1.0)
    assert not torch.equal(q, result.applied_step)


@pytest.mark.parametrize(
    "displacement,updates",
    [(0.0, False), (0.999e-7, False), (1e-7, True), (1.001e-7, True)],
)
def test_ase_displacement_guard(displacement, updates):
    state = bfgs_ase_prepare_state(3, alpha=7.0)
    state.initialized, state.step_count = True, 1
    q = torch.tensor([[displacement, 0.0, 0.0]], dtype=torch.float64)
    f = -2.0 * q
    initial = state.hessian.clone()
    atoms = Atoms("H", positions=q.numpy())
    oracle = BFGS(atoms, logfile=None, alpha=7.0)
    oracle.update(np.zeros(3), np.zeros(3), None, None)
    oracle.update(q.numpy().ravel(), f.numpy().ravel(), np.zeros(3), np.zeros(3))
    bfgs_ase_step(q, f, state)
    torch.testing.assert_close(
        state.hessian, torch.tensor(oracle.H), rtol=1e-10, atol=1e-12
    )
    assert (not torch.equal(initial, state.hessian)) == updates


@pytest.mark.parametrize(
    "diagonal", [[-2.0, 3.0, 4.0], [2.0, 2.0, 2.0], [1e-16, 3.0, 4.0]]
)
def test_absolute_spectrum_without_clamping(diagonal):
    state = bfgs_ase_prepare_state(3, maxstep=4.0)
    state.initialized, state.step_count = True, 1
    state.hessian.copy_(torch.diag(torch.tensor(diagonal, dtype=torch.float64)))
    q = torch.zeros((1, 3), dtype=torch.float64)
    f = torch.tensor([diagonal], dtype=q.dtype)
    result = bfgs_ase_step(q, f, state)
    torch.testing.assert_close(result.raw_step, torch.sign(f), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("split", [0, 1, 5])
def test_restart_matches_uninterrupted_and_serializes(split):
    q = torch.tensor([[1.3, -0.7, 0.8], [-0.6, 1.1, -0.9]], dtype=torch.float64)
    reference_q = q.clone()
    state = bfgs_ase_prepare_state(6, system_id=8)
    reference = bfgs_ase_prepare_state(6, system_id=8)
    for step in range(8):
        if step == split:
            buffer = io.BytesIO()
            torch.save(state.state_dict(), buffer)
            buffer.seek(0)
            restored = ASEBFGSState.from_state_dict(
                torch.load(buffer, weights_only=True)
            )
            assert restored.hessian.data_ptr() != state.hessian.data_ptr()
            assert_snapshot_equal(restored, state.state_dict())
            state = restored
        bfgs_ase_step(q, torch.tensor(potential_force(q.numpy(), "quartic")), state)
        bfgs_ase_step(
            reference_q,
            torch.tensor(potential_force(reference_q.numpy(), "quartic")),
            reference,
        )
    torch.testing.assert_close(q, reference_q, rtol=0, atol=0)
    assert_snapshot_equal(state, reference.state_dict())


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 99),
        ("algorithm", "LBFGS"),
        ("num_scalar_dofs", 6),
        ("alpha", 0.0),
        ("maxstep", float("nan")),
        ("system_id", True),
        ("step_count", -1),
        ("initialized", True),
        ("hessian", torch.eye(3)),
        ("previous_positions", torch.zeros(2, 3, dtype=torch.float64)),
    ],
)
def test_invalid_restart_rejected(key, value):
    state = bfgs_ase_prepare_state(3)
    original = state.state_dict()
    snapshot = state.state_dict()
    snapshot[key] = value
    with pytest.raises((TypeError, ValueError)):
        ASEBFGSState.from_state_dict(snapshot)
    assert_snapshot_equal(state, original)


def test_snapshot_missing_key_nonfinite_and_no_alias():
    state = bfgs_ase_prepare_state(3)
    snapshot = state.state_dict()
    snapshot["hessian"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ASEBFGSState.from_state_dict(snapshot)
    assert torch.isfinite(state.hessian).all()
    snapshot = state.state_dict()
    del snapshot["maxstep"]
    with pytest.raises(ValueError, match="keys"):
        ASEBFGSState.from_state_dict(snapshot)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_scalar_dofs": 0},
        {"num_scalar_dofs": 4},
        {"num_scalar_dofs": True},
        {"alpha": 0.0},
        {"alpha": float("inf")},
        {"maxstep": -0.2},
        {"maxstep": True},
        {"dtype": torch.float32},
        {"system_id": "x"},
        {"device": "meta"},
    ],
)
def test_prepare_rejects_unsupported_domain(kwargs):
    args = {"num_scalar_dofs": 3, **kwargs}
    with pytest.raises((TypeError, ValueError, BackendUnavailableError)):
        bfgs_ase_prepare_state(**args)


@pytest.mark.parametrize(
    "bad",
    [
        "shape",
        "dtype",
        "stride",
        "alias",
        "history_alias",
        "nonfinite",
        "grad",
        "identity",
        "history_device",
    ],
)
def test_step_validation_is_atomic(bad):
    state = bfgs_ase_prepare_state(6, system_id=1)
    q = torch.zeros((2, 3), dtype=torch.float64)
    f = torch.ones_like(q)
    kwargs = {}
    if bad == "shape":
        f = f.reshape(3, 2)
    elif bad == "dtype":
        f = f.float()
    elif bad == "stride":
        f = torch.ones((3, 2), dtype=q.dtype).T
    elif bad == "alias":
        f = q
    elif bad == "history_alias":
        q = state.previous_positions
    elif bad == "nonfinite":
        f[0, 0] = float("nan")
    elif bad == "grad":
        f.requires_grad_(True)
    elif bad == "identity":
        kwargs["system_id"] = 2
    elif bad == "history_device":
        state.previous_forces = torch.zeros((2, 3), dtype=q.dtype, device="meta")
    original_q = q.clone()
    original_h = state.hessian.clone()
    with pytest.raises((TypeError, ValueError)):
        bfgs_ase_step(q, f, state, **kwargs)
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
    torch.testing.assert_close(state.hessian, original_h, rtol=0, atol=0)
    assert state.step_count == 0


@pytest.mark.parametrize(
    "failure", ["curvature", "hessian_denominator", "spectrum", "overflow"]
)
def test_numerical_failure_before_mutation(failure):
    state = bfgs_ase_prepare_state(3)
    state.initialized, state.step_count = True, 1
    q = torch.zeros((1, 3), dtype=torch.float64)
    f = torch.ones_like(q)
    if failure == "curvature":
        q[0, 0] = 1.0
        state.previous_forces.copy_(f)
    elif failure == "hessian_denominator":
        state.hessian.copy_(torch.diag(torch.tensor([1.0, -1.0, 1.0])))
        q[0, :2] = 1.0
    elif failure == "spectrum":
        state.hessian[0, 0] = 0.0
    else:
        state.hessian.mul_(1e-300)
        f.mul_(1e100)
    before_q, snapshot = q.clone(), state.state_dict()
    with pytest.raises(BFGSNumericalError):
        bfgs_ase_step(q, f, state)
    torch.testing.assert_close(q, before_q, rtol=0, atol=0)
    assert_snapshot_equal(state, snapshot)


@pytest.mark.parametrize(
    "backend", [None, "warp", "auto", "hip", "triton", "does_not_exist"]
)
def test_unavailable_backend_is_explicit(backend):
    message = "upstream has no legacy BFGS" if backend in {None, "warp"} else None
    with pytest.raises(BackendUnavailableError, match=message):
        bfgs_ase_prepare_state(3, backend=backend)


@pytest.mark.parametrize(
    "features,gradient_order",
    [
        ({"fixed_cell", "ragged_kernel"}, 0),
        ({"variable_cell", "six_cell_dofs"}, 0),
        ({"fixed_cell"}, 1),
    ],
)
def test_catalog_does_not_admit_future_capabilities(features, gradient_order):
    with pytest.raises(BackendUnavailableError):
        resolve_backend(
            "torch_reference",
            operation="bfgs_ase_compat",
            device="cpu",
            dtype=torch.float64,
            features=features,
            gradient_order=gradient_order,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "fire"},
        {"dtype": "float32"},
        {"device": "cuda"},
        {"features": ("variable_cell",)},
        {"gradient_order": 1},
    ],
)
def test_cached_selection_cannot_bypass_contract(change):
    selection = resolve_backend(
        "torch_reference",
        operation="bfgs_ase_compat",
        device="cpu",
        dtype=torch.float64,
        features={"fixed_cell", "single_system"},
    )
    with pytest.raises(BackendUnavailableError):
        bfgs_ase_prepare_state(3, selection=dataclasses.replace(selection, **change))


def test_second_metadata_executor_needs_no_dispatch_table(monkeypatch):
    """A fixture family exercises the existing generic executor with real calls."""
    from nvalchemiops import bfgs_ase_reference as reference

    calls = []
    module = types.ModuleType("bfgs_fixture_executor")

    def prepare(*args, **kwargs):
        calls.append("prepare")
        return reference.bfgs_ase_prepare_state(*args, **kwargs)

    def step(*args, **kwargs):
        calls.append("step")
        return reference.bfgs_ase_step(*args, **kwargs)

    module.bfgs_ase_prepare_state = prepare
    module.bfgs_ase_step = step
    impl = Implementation(
        implementation_id="fixture.bfgs",
        operation="bfgs_ase_compat",
        family="fixture",
        executor=module.__name__,
        entrypoints=("bfgs_ase_prepare_state", "bfgs_ase_step"),
        executor_owner="ops",
        features=frozenset({"single_system", "fixed_cell"}),
        dtypes=frozenset({"float64"}),
        default_strategy=True,
        auto_enabled=False,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        registry_api,
        "DEFAULT_IMPLEMENTATION_REGISTRY",
        registry_api.ImplementationRegistry([impl]),
    )
    clear_entrypoint_cache()
    try:
        state = bfgs_ase_prepare_state(3, backend="fixture")
        q = torch.zeros((1, 3), dtype=torch.float64)
        bfgs_ase_step(q, torch.ones_like(q), state, backend="fixture")
        assert calls == ["prepare", "step"]
        assert state.step_count == 1 and q.norm() > 0
    finally:
        clear_entrypoint_cache()


def test_import_and_execution_without_warp_ase_framework():
    """Block optional dependencies in a fresh interpreter, then execute BFGS."""
    code = """
import sys
import torch
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'warp', 'ase', 'nvalchemi'}:
            raise ImportError('blocked: ' + fullname)
sys.meta_path.insert(0, Block())
from nvalchemiops.bfgs_ase import bfgs_ase_prepare_state, bfgs_ase_step
q = torch.zeros((1, 3), dtype=torch.float64)
state = bfgs_ase_prepare_state(3)
bfgs_ase_step(q, torch.ones_like(q), state)
assert state.step_count == 1
assert not any(name.split('.')[0] in {'warp', 'ase', 'nvalchemi'} for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and literal test code
        [sys.executable, "-c", code],
        env={**os.environ, "HIP_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
