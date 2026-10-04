"""NHC public API, upstream bug regressions and framework lifecycle."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch
from nvalchemiops.backend import BackendUnavailableError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTNoseHoover
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def batch(sizes=(2, 4), device="cpu", dtype=torch.float64):
    systems = []
    for j, n in enumerate(sizes):
        p = torch.arange(n * 3, dtype=dtype).reshape(n, 3) * 0.005 + 0.05
        data = AtomicData(
            positions=p,
            atomic_numbers=torch.ones(n, dtype=torch.long),
            atomic_masses=torch.ones(n, dtype=dtype),
            forces=-p,
            energy=torch.zeros(1, 1, dtype=dtype),
            cell=(torch.eye(3, dtype=dtype) * 5)[None],
            pbc=torch.ones(1, 3, dtype=torch.bool),
        )
        data.add_node_property("velocities", torch.full_like(p, 0.01 * (j + 1)))
        systems.append(data)
    return Batch.from_data_list(systems).to(device)


def dynamics(**kwargs):
    model = kwargs.pop("model", DemoModelWrapper(DemoModel()).double())
    return NVTNoseHoover(
        model,
        dt=kwargs.pop("dt", 0.1),
        temperature=kwargs.pop("temperature", 300.0),
        thermostat_time=kwargs.pop("thermostat_time", 10.0),
        backend=kwargs.pop("backend", "torch_reference"),
        **kwargs,
    )


def test_import_and_public_run_without_warp():
    code = "from nvalchemi.dynamics import NVTNoseHoover; import sys; assert 'warp' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())  # noqa: S603 - fixed import-only script
    b = batch()
    cell = b.cell.clone()
    d = dynamics()
    d.run(b, n_steps=3)
    assert d.step_count == 3 and b.batch_ptr.tolist() == [0, 2, 6]
    assert torch.equal(b.cell, cell) and torch.isfinite(b.velocities).all()


@pytest.mark.parametrize("order", [3, 5])
def test_yoshida_order_forwarded_to_actual_executor(order, monkeypatch):
    from nvalchemiops.executor import clear_entrypoint_cache

    import nvalchemi._backend as backend_module
    import nvalchemi._dynamics_reference.nose_hoover as executor

    clear_entrypoint_cache()
    calls = []
    original = executor.nhc_chain_update

    def spy(*args, **kwargs):
        calls.append(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "nhc_chain_update", spy)
    b = batch()
    d = dynamics(yoshida_order=order)
    d._init_state(b)
    monkeypatch.setattr(
        backend_module,
        "resolve_backend",
        lambda *a, **k: pytest.fail("selection resolved twice"),
    )
    d.pre_update(b)
    d.post_update(b)
    assert calls == [order, order]
    clear_entrypoint_cache()


def test_initial_and_refill_chain_masses_use_actual_counts():
    b = batch()
    d = dynamics()
    d._init_state(b)
    d._state.nhc_eta[1].fill_(0.7)
    old = d._state.index_select(torch.tensor([1])).clone()
    updated = batch((4, 7, 3))
    d._sync_state_to_batch(torch.tensor([1]), 2, updated)
    torch.testing.assert_close(d._state.nhc_eta[0], old.nhc_eta[0])
    assert d._state.nhc_ndof.tolist() == [12.0, 21.0, 9.0]
    base = d._state.temperature * d._state.thermostat_time.square()
    torch.testing.assert_close(d._state.nhc_Q[:, 0], base * d._state.nhc_ndof)
    torch.testing.assert_close(d._state.nhc_Q[:, 1], base)
    assert not d._state.nhc_eta[1:].any()


def test_heterogeneous_parameters_and_ambiguous_refill_rejection():
    b = batch()
    d = dynamics(
        temperature=torch.tensor([300.0, 600.0]),
        dt=torch.tensor([0.1, 0.2]),
        thermostat_time=torch.tensor([10.0, 20.0]),
    )
    d._init_state(b)
    torch.testing.assert_close(
        d._state.temperature,
        torch.tensor([300.0, 600.0], dtype=b.positions.dtype) * 8.617333262145e-5,
    )
    with pytest.raises(ValueError, match="scalar constructor"):
        d._make_new_state(1, batch((4, 7)))


@pytest.mark.parametrize(
    "method", ["_masked_pre_update", "_masked_post_update", "masked_update", "step"]
)
def test_inactive_graph_chain_does_not_advance(method):
    b = batch()
    d = dynamics()
    d._init_state(b)
    initial = d._state.clone()
    pos = b.positions.clone()
    vel = b.velocities.clone()
    if method == "step":
        b["status"] = torch.tensor([[0], [d.exit_status]], dtype=torch.int32)
        d.step(b)
    else:
        getattr(d, method)(b, torch.tensor([True, False]))
    for key in (
        "nhc_eta",
        "nhc_eta_dot",
        "nhc_ke2",
        "nhc_total_scale",
        "nhc_step_scale",
        "nhc_dt_chain",
    ):
        torch.testing.assert_close(
            getattr(d._state, key)[1], getattr(initial, key)[1], rtol=0, atol=0
        )
    torch.testing.assert_close(b.positions[2:], pos[2:], rtol=0, atol=0)
    torch.testing.assert_close(b.velocities[2:], vel[2:], rtol=0, atol=0)
    assert d._state.nhc_eta_dot[0].abs().sum() > 0


def test_restart_split_matches_continuous_and_rejects_invalid_state():
    b = batch()
    d = dynamics()
    d.run(b, n_steps=3)
    snapshot = b.clone()
    saved = d.state_dict()
    d.run(b, n_steps=2)
    resumed = dynamics(model=d.model)
    resumed.load_state_dict(saved)
    resumed.run(snapshot, n_steps=2)
    for key in ("positions", "velocities", "forces", "energy"):
        torch.testing.assert_close(
            getattr(b, key), getattr(snapshot, key), rtol=0, atol=0
        )
    for key in d._restart_keys:
        torch.testing.assert_close(
            getattr(d._state, key), getattr(resumed._state, key), rtol=0, atol=0
        )
    assert resumed.step_count == d.step_count == 5
    broken = dict(saved, version=99)
    with pytest.raises(ValueError, match="version"):
        resumed.load_state_dict(broken)
    assert resumed.step_count == 5
    wrong = dynamics(model=d.model)
    wrong.load_state_dict(saved)
    for _ in range(2):
        with pytest.raises(ValueError, match="atom counts"):
            wrong._ensure_state_initialized(batch((3, 3)))
        assert not hasattr(wrong, "_state")
    wrong._ensure_state_initialized(batch())
    assert wrong.step_count == 3


def test_explicit_backend_and_constructor_failures():
    for backend in ("hip", "triton", "unknown"):
        with pytest.raises(BackendUnavailableError):
            dynamics(backend=backend)._init_state(batch())
    for kwargs in (
        {"chain_length": 0},
        {"chain_length": 9},
        {"yoshida_order": 1},
        {"yoshida_order": 3.0},
    ):
        with pytest.raises(ValueError):
            dynamics(**kwargs)
    for kwargs in ({"temperature": 0.0}, {"thermostat_time": 0.0}, {"dt": -0.1}):
        with pytest.raises(ValueError, match="positive"):
            dynamics(**kwargs)._init_state(batch())


def test_custom_op_mutation_fake_opcheck_and_eager_compile():
    from nvalchemi._dynamics_reference.nose_hoover import nhc_chain_update

    b = batch()
    d = dynamics()
    d._init_state(b)
    s = d._state
    args = (
        b.velocities,
        b.atomic_masses,
        s.nhc_eta,
        s.nhc_eta_dot,
        s.nhc_Q,
        s.temperature,
        s.dt,
        s.nhc_ndof,
        s.nhc_ke2,
        s.nhc_total_scale,
        s.nhc_step_scale,
        s.nhc_dt_chain,
        b.batch_idx.int(),
        True,
        3,
    )
    result = torch.library.opcheck(nhc_chain_update, args)
    assert all(value == "SUCCESS" for value in result.values())
    copies = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
    fn = torch.compile(nhc_chain_update, backend="eager", fullgraph=True)
    fn(*copies)
    nhc_chain_update(*args)
    for a, c in zip(args, copies):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, c)


def test_hooks_preserve_all_step_stages():
    from nvalchemi.dynamics.base import DynamicsStage

    calls = []

    class Observer:
        frequency = 1
        stage = DynamicsStage.BEFORE_STEP

        def _runs_on_stage(self, stage):
            return True

        def __call__(self, ctx, stage):
            calls.append(stage)

    d = dynamics(hooks=[Observer()])
    d.run(batch(), n_steps=1)
    expected = [
        DynamicsStage.BEFORE_STEP,
        DynamicsStage.BEFORE_PRE_UPDATE,
        DynamicsStage.AFTER_PRE_UPDATE,
        DynamicsStage.BEFORE_COMPUTE,
        DynamicsStage.AFTER_COMPUTE,
        DynamicsStage.BEFORE_POST_UPDATE,
        DynamicsStage.AFTER_POST_UPDATE,
        DynamicsStage.AFTER_STEP,
    ]
    assert calls == expected


def test_fusedstage_matches_separate_graph_runs_and_migrates():
    from nvalchemi.dynamics.base import ConvergenceHook, FusedStage

    b = batch()
    model = DemoModelWrapper(DemoModel()).double()
    dynamics(model=model).compute(b)
    reference = [b.index_select(torch.tensor([i])) for i in range(2)]
    d0 = dynamics(model=model, temperature=300.0)
    d1 = dynamics(model=model, temperature=600.0)
    r0 = dynamics(model=model, temperature=300.0)
    r1 = dynamics(model=model, temperature=600.0)
    b["status"] = torch.tensor([[0], [1]], dtype=torch.long)
    fused = FusedStage(sub_stages=[(0, d0), (1, d1)], n_steps=1)
    fused.step(b)
    r0.step(reference[0])
    r1.step(reference[1])
    for i, ref in enumerate(reference):
        sl = slice(int(b.batch_ptr[i]), int(b.batch_ptr[i + 1]))
        torch.testing.assert_close(b.positions[sl], ref.positions)
        torch.testing.assert_close(b.velocities[sl], ref.velocities)
    assert not d0._state.nhc_eta[1].any() and not d1._state.nhc_eta[0].any()
    hook = ConvergenceHook.from_fmax(1e6, source_status=0, target_status=1)
    d0.register_hook(hook)
    fused.step(b)
    assert b.status.reshape(-1).tolist() == [1, 1]
    saved = d0._state.nhc_eta.clone()
    fused.step(b)
    torch.testing.assert_close(d0._state.nhc_eta, saved, rtol=0, atol=0)


def test_actual_sampler_refill_preserves_ids_and_chain_state():
    from nvalchemi.dynamics.sampler import SizeAwareSampler

    class Dataset:
        sizes = (2, 4, 3, 7)

        def __len__(self):
            return len(self.sizes)

        def get_metadata(self, i):
            return self.sizes[i], 0

        def __getitem__(self, i):
            return batch((self.sizes[i],)).to_data_list()[0], {}

    sampler = SizeAwareSampler(Dataset(), max_batch_size=2)
    b = sampler.build_initial_batch()
    d = dynamics(sampler=sampler)
    d._init_state(b)
    d.pre_update(b)
    old = d._state.clone()
    old_id = b.system_id[1].clone()
    b["status"] = torch.tensor([[1], [0]], dtype=torch.long)
    updated = d.refill_check(b, exit_status=1)
    assert updated is not None and updated.num_graphs == 2
    assert torch.equal(updated.system_id[0], old_id)
    torch.testing.assert_close(d._state.nhc_eta[0], old.nhc_eta[1])
    counts = updated.batch_ptr[1:] - updated.batch_ptr[:-1]
    torch.testing.assert_close(d._state.nhc_ndof, counts.to(d._state.dt.dtype) * 3)
    torch.testing.assert_close(
        d._state.nhc_Q[:, 0],
        d._state.temperature * d._state.thermostat_time.square() * counts * 3,
    )
    d.step(updated)


def test_domainparallel_explicit_reference_rejected_before_collective():
    # Load only the adapter function: importing the distributed package requires
    # optional PhysicsNeMo, which is not installed in the reference environment.
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import nvalchemi

    path = Path(nvalchemi.__file__).parent / "distributed/_dynamics_coordinator.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_make_global_nhc_chain_update"
    )
    scope = {
        "ParallelizationStrategy": object,
        "torch": torch,
        "Reduce": SimpleNamespace(SUM=None),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 - trusted local adapter AST
    _make_global_nhc_chain_update = scope["_make_global_nhc_chain_update"]
    from nvalchemiops.backend import resolve_backend

    strategy = Mock()
    operation = _make_global_nhc_chain_update(strategy)
    selection = resolve_backend(
        "torch_reference",
        operation="nose_hoover",
        device="cpu",
        dtype=torch.float64,
        features={"fixed_cell"},
    )
    with pytest.raises(BackendUnavailableError, match="DomainParallel"):
        operation(*([None] * 13), selection=selection, nloops=3)
    strategy.reduce_system.assert_not_called()


def test_registered_alternative_executor_uses_same_dispatcher(monkeypatch):
    from dataclasses import replace
    from types import ModuleType

    import nvalchemiops.backend as registry_module
    from nvalchemiops.executor import clear_entrypoint_cache

    from nvalchemi.dynamics._ops.nose_hoover import nhc_compute_masses

    catalog = registry_module.backend_capabilities()
    source = next(x for x in catalog if x.operation == "nose_hoover")
    alternative = replace(
        source, implementation_id="test.nhc-v1", executor="test_nhc_executor"
    )
    module = ModuleType("test_nhc_executor")
    calls = []

    def compute(*args):
        calls.append(args)
        return torch.ones(2, 3, dtype=torch.float64)

    module.nhc_compute_masses = compute
    monkeypatch.setitem(sys.modules, "test_nhc_executor", module)
    monkeypatch.setattr(
        registry_module,
        "DEFAULT_IMPLEMENTATION_REGISTRY",
        registry_module.ImplementationRegistry([*catalog, alternative]),
    )
    b = batch()
    d = dynamics()
    d._init_state(b)
    s = d._state
    result = nhc_compute_masses(
        s.temperature,
        s.thermostat_time,
        b.atomic_masses,
        b.batch_idx.int(),
        3,
        backend="test.nhc-v1",
    )
    assert len(calls) == 1 and result.shape == (2, 3)
    clear_entrypoint_cache()


def test_alias_rejection_and_repeated_scale_reset():
    from nvalchemi._dynamics_reference.nose_hoover import nhc_chain_update

    b = batch()
    d = dynamics()
    d._init_state(b)
    s = d._state

    def args():
        return (
            b.velocities,
            b.atomic_masses,
            s.nhc_eta,
            s.nhc_eta_dot,
            s.nhc_Q,
            s.temperature,
            s.dt,
            s.nhc_ndof,
            s.nhc_ke2,
            s.nhc_total_scale,
            s.nhc_step_scale,
            s.nhc_dt_chain,
            b.batch_idx.int(),
            True,
            3,
        )

    bad = list(args())
    bad[10] = bad[9]
    with pytest.raises(ValueError, match="alias"):
        nhc_chain_update(*bad)
    copies = [x.clone() if isinstance(x, torch.Tensor) else x for x in args()]
    s.nhc_total_scale.fill_(999.0)
    nhc_chain_update(*args())
    nhc_chain_update(*copies)
    for a, c in zip(args(), copies):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, c)


def test_after_step_hook_observes_restored_inactive_chain():
    from nvalchemi.dynamics.base import DynamicsStage

    b = batch()
    b["status"] = torch.tensor([[0], [1]])
    observed = []

    class Observer:
        frequency = 1
        stage = DynamicsStage.AFTER_STEP

        def __call__(self, ctx, stage):
            observed.append(ctx.workflow._state.nhc_eta[1].clone())

    d = dynamics(hooks=[Observer()])
    d.step(b)
    assert len(observed) == 1 and not observed[0].any()
