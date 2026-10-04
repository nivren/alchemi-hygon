# SPDX-License-Identifier: Apache-2.0
"""Public variable-cell optimizers and frame-consistent alignment hook."""

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from nvalchemiops.backend import (
    BackendUnavailableError,
    Implementation,
    ImplementationRegistry,
)
from nvalchemiops.executor import clear_entrypoint_cache

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.models.base import BaseModelMixin, ModelConfig


class CellHarmonic(BaseModelMixin):
    """Analytic E(H) with exact row-cell stress, zero Cartesian forces."""

    def __init__(self):
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}), supports_pbc=True
        )

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        return data

    def __call__(self, batch):
        h = batch.cell
        target = torch.eye(3, dtype=h.dtype, device=h.device).expand_as(h) * 2
        gradient = h - target
        volume = torch.linalg.det(h).abs()
        return {
            "energy": 0.5 * gradient.square().sum((-1, -2)),
            "forces": torch.zeros_like(batch.positions),
            "stress": h.mT @ gradient / volume[:, None, None],
        }


def batch(dtype=torch.float64, device="cpu"):
    systems = []
    for i, n in enumerate((2, 3)):
        h = torch.tensor(
            [[2.2, 0, 0], [0.1, 2.1, 0], [0.03, -0.02, 2.3]], dtype=dtype
        ) * (1 + 0.05 * i)
        r = torch.arange(n * 3, dtype=dtype).reshape(n, 3) / 11 @ h
        systems.append(
            AtomicData(
                positions=r,
                atomic_numbers=torch.ones(n, dtype=torch.long),
                atomic_masses=torch.ones(n, dtype=dtype),
                velocities=torch.zeros_like(r),
                forces=torch.zeros_like(r),
                stress=torch.zeros(1, 3, 3, dtype=dtype),
                energy=torch.zeros(1, 1, dtype=dtype),
                cell=h[None],
                pbc=torch.ones(1, 3, dtype=torch.bool),
            )
        )
    return Batch.from_data_list(systems).to(device)


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_public_stress_drives_cell_and_short_analytic_convergence(cls):
    b = batch()
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d.compute(b)
    start = CellHarmonic()(b)["energy"].clone()
    old = b.cell.clone()
    frac = torch.linalg.solve(
        b.cell[b.batch_idx.long()].mT, b.positions[..., None]
    ).squeeze(-1)
    d.run(b, n_steps=160)
    assert d.step_count == 160 and not torch.equal(b.cell, old)
    assert torch.all(b.energy.flatten() < start * 1e-3)
    torch.testing.assert_close(
        torch.linalg.solve(
            b.cell[b.batch_idx.long()].mT, b.positions[..., None]
        ).squeeze(-1),
        frac,
        rtol=2e-11,
        atol=2e-11,
    )
    assert "warp" not in sys.modules


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_selections_frozen_after_initialization(cls, monkeypatch):
    import nvalchemi._backend as module

    b = batch()
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d._init_state(b)
    d.compute(b)
    monkeypatch.setattr(
        module, "resolve_backend", lambda *a, **k: pytest.fail("resolved again")
    )
    d.pre_update(b)
    d.compute(b)
    d.post_update(b)


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
@pytest.mark.parametrize("backend", ["hip", "triton", "not-a-backend"])
def test_unsupported_backend_fails(cls, backend):
    with pytest.raises(BackendUnavailableError):
        cls(model=CellHarmonic(), dt=0.04, backend=backend)._init_state(batch())


@pytest.mark.parametrize("mask_kind", ["nonperiodic", "inactive"])
def test_alignment_hook_periodic_subset_covariance_and_one_time_behavior(mask_kind):
    b = batch()
    rotation = torch.tensor(
        [[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1]], dtype=b.positions.dtype
    )
    b.cell.copy_(b.cell @ rotation)
    b.positions.copy_(b.positions @ rotation)
    workflow = None
    if mask_kind == "nonperiodic":
        b.pbc[1].zero_()
    else:
        b.status = torch.tensor([[0], [1]], dtype=torch.int32)
        workflow = SimpleNamespace(exit_status=1)
    b.forces = torch.arange(15, dtype=b.positions.dtype).reshape(5, 3) / 10
    b.velocities.copy_(b.forces * 0.2)
    b.stress = torch.tensor(
        [[[0.1, 0.02, 0.03], [0.02, 0.2, -0.01], [0.03, -0.01, 0.3]]] * 2,
        dtype=b.cell.dtype,
    )
    old = {
        k: getattr(b, k).clone()
        for k in ("positions", "cell", "forces", "velocities", "stress")
    }
    hook = AlignCellHook(backend="torch_reference")
    ctx = DynamicsContext(batch=b, workflow=workflow)
    hook(ctx, hook.stage)
    t = torch.linalg.solve(b.cell[0], old["cell"][0])
    torch.testing.assert_close(b.forces[:2], old["forces"][:2] @ t.mT)
    torch.testing.assert_close(b.velocities[:2], old["velocities"][:2] @ t.mT)
    torch.testing.assert_close(b.stress[0], t @ old["stress"][0] @ t.mT)
    for key in ("cell", "stress"):
        assert torch.equal(getattr(b, key)[1], old[key][1])
    for key in ("positions", "forces", "velocities"):
        assert torch.equal(getattr(b, key)[2:], old[key][2:])
    again = {k: getattr(b, k).clone() for k in old}
    hook(ctx, hook.stage)
    assert all(torch.equal(getattr(b, k), v) for k, v in again.items())


def test_alignment_hook_inherits_frozen_optimizer_selection(monkeypatch):
    import nvalchemi._backend as module

    b = batch()
    b.cell[:, 0, 1] = 0.2
    d = FIRE2VariableCell(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d._init_state(b)
    hook = AlignCellHook()
    monkeypatch.setattr(
        module, "resolve_backend", lambda *a, **k: pytest.fail("resolved again")
    )
    hook(DynamicsContext(batch=b, workflow=d), hook.stage)
    assert hook._selection is d._cell_tools_selection
    assert not torch.count_nonzero(torch.triu(b.cell, 1))


def test_public_import_is_warp_free():
    script = "from nvalchemi.dynamics import FIREVariableCell,FIRE2VariableCell; import sys; assert 'warp' not in sys.modules"
    result = subprocess.run(  # noqa: S603 - fixed import script
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cls", [FIRE2VariableCell, FIREVariableCell])
def test_alignment_rejects_reorientation_after_cell_motion_started(cls):
    b = batch()
    b.cell[:, 0, 1] = 0.2
    d = cls(model=CellHarmonic(), dt=0.04, backend="torch_reference")
    d._init_state(b)
    name = "cell_velocities" if cls is FIRE2VariableCell else "cell_velocity"
    getattr(d._state, name).fill_(0.1)
    old_h, old_r = b.cell.clone(), b.positions.clone()
    with pytest.raises(ValueError, match="before initializing cell motion"):
        AlignCellHook()(DynamicsContext(batch=b, workflow=d), DynamicsStage.BEFORE_STEP)
    assert torch.equal(old_h, b.cell) and torch.equal(old_r, b.positions)


def test_second_executor_uses_unchanged_dispatchers(monkeypatch):
    import nvalchemiops.backend as backend_module

    from nvalchemi.dynamics._ops.fire import fire2_step_coord_cell
    from nvalchemi.dynamics._ops.fire_cell import fire_cell_post, fire_cell_pre

    module = ModuleType("variable_cell_second_executor")
    calls = []
    for name in ("fire2_step_coord_cell", "fire_cell_pre", "fire_cell_post"):

        def entry(*args, _name=name, **kwargs):
            calls.append((_name, len(args)))

        setattr(module, name, entry)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = ImplementationRegistry(
        backend_module.DEFAULT_IMPLEMENTATION_REGISTRY.implementations()
    )
    for operation, names, features in (
        ("fire2", ("fire2_step_coord_cell",), {"variable_cell"}),
        (
            "fire_cell",
            ("fire_cell_pre", "fire_cell_post"),
            {"variable_cell", "full_cell"},
        ),
    ):
        registry.register(
            Implementation(
                implementation_id=f"test.{operation}-cell-v1",
                operation=operation,
                family="test",
                executor=module.__name__,
                entrypoints=names,
                executor_owner="ops",
                features=frozenset(features),
                max_gradient_order=0,
                default_strategy=True,
            )
        )
    monkeypatch.setattr(backend_module, "DEFAULT_IMPLEMENTATION_REGISTRY", registry)
    clear_entrypoint_cache()
    try:
        r = torch.zeros(2, 3)
        h = torch.eye(3)[None]
        idx = torch.zeros(2, dtype=torch.int32)
        a = torch.ones(1) * 0.1
        dt = torch.ones(1) * 0.01
        n = torch.zeros(1, dtype=torch.int32)
        fire2_step_coord_cell(
            r,
            r.clone(),
            r.clone(),
            h,
            h.clone(),
            h.clone(),
            idx,
            a,
            dt,
            n,
            backend="test",
        )
        fire_cell_pre(
            r,
            r.clone(),
            r.clone(),
            torch.ones(2),
            h,
            h.clone(),
            h.clone(),
            dt,
            idx,
            backend="test",
        )
        fire_cell_post(
            r,
            r.clone(),
            torch.ones(2),
            h.clone(),
            h.clone(),
            idx,
            a,
            dt,
            n,
            a.clone(),
            a.clone(),
            dt.clone(),
            dt.clone(),
            n.clone(),
            a.clone(),
            a.clone(),
            backend="test",
        )
        assert calls == [
            ("fire2_step_coord_cell", 10),
            ("fire_cell_pre", 9),
            ("fire_cell_post", 16),
        ]
    finally:
        clear_entrypoint_cache()


def test_default_full_fire_stays_on_legacy_handlers(monkeypatch):
    import nvalchemi.dynamics._ops.fire_cell as module

    calls = []
    monkeypatch.setattr(module, "_legacy_pre", lambda *a, **k: calls.append("pre"))
    monkeypatch.setattr(module, "_legacy_post", lambda *a, **k: calls.append("post"))
    for backend in (None, "warp"):
        b = batch()
        d = FIREVariableCell(model=CellHarmonic(), dt=0.04, backend=backend)
        d._init_state(b)
        # Isolate the default dispatch guard from missing actual Warp runtime.
        monkeypatch.setattr(d, "_stress_selection", None)
        import nvalchemi.dynamics._ops.stress as stress

        monkeypatch.setattr(
            stress, "stress_to_cell_force", lambda *a, **k: torch.zeros_like(b.cell)
        )
        d.pre_update(b)
        d.post_update(b)
    assert calls == ["pre", "post", "pre", "post"]
