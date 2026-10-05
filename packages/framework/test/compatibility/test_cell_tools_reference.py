# SPDX-License-Identifier: Apache-2.0
"""Existing alignment API, lazy legacy bridge and declarative executor ABI."""

import subprocess
import sys
from types import ModuleType

import pytest
import torch
from nvalchemiops.backend import (
    DEFAULT_IMPLEMENTATION_REGISTRY,
    BackendUnavailableError,
    Implementation,
    ImplementationRegistry,
    resolve_backend,
)
from nvalchemiops.executor import clear_entrypoint_cache, execute_selected

from nvalchemi.dynamics._ops.cell_align import align_cell


def tensors():
    h = torch.tensor(
        [[[2.0, 0.3, -0.2], [0.5, 2.7, 0.1], [0.3, -0.4, 3.1]]], dtype=torch.float64
    )
    r = torch.tensor([[0.1, 0.2, 0.3], [0.8, -0.1, 1.2]], dtype=h.dtype) @ h[0]
    return r, h


def test_import_and_reference_without_warp():
    code = """
import sys, torch
from nvalchemi.dynamics._ops.cell_align import align_cell
h = torch.eye(3, dtype=torch.float64)[None]
r = torch.ones(2,3,dtype=torch.float64)
assert align_cell(r,h,backend='torch_reference') is None
assert 'warp' not in sys.modules
assert hasattr(torch.ops.nvalchemi, 'align_cell')
"""
    result = subprocess.run(  # noqa: S603 - fixed import-only script
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_pre_resolved_selection_not_resolved_twice(monkeypatch):
    import nvalchemi._backend as module

    r, h = tensors()
    frac = r @ torch.linalg.inv(h[0])
    selected = resolve_backend(
        "torch_reference",
        operation="cell_tools",
        device="cpu",
        dtype=h.dtype,
        gradient_order=0,
        features={"row_cell"},
    )
    monkeypatch.setattr(
        module, "resolve_backend", lambda *a, **k: pytest.fail("resolved twice")
    )
    assert align_cell(r, h, selection=selected) is None
    torch.testing.assert_close(r @ torch.linalg.inv(h[0]), frac)


@pytest.mark.parametrize("backend", ["hip", "triton", "not-a-backend"])
def test_unsupported_backend(backend):
    r, h = tensors()
    with pytest.raises(BackendUnavailableError):
        align_cell(r, h, backend=backend)


def test_gradient_and_variable_cell_capabilities_not_advertised():
    for kwargs in ({"gradient_order": 1}, {"features": {"variable_cell"}}):
        with pytest.raises(BackendUnavailableError):
            resolve_backend(
                "torch_reference", operation="cell_tools", device="cpu", **kwargs
            )
    r, h = tensors()
    with pytest.raises(ValueError, match="must target operation"):
        align_cell(r, h, selection=resolve_backend(None, operation="fire2"))


def test_legacy_selection_preserved(monkeypatch):
    import nvalchemi.dynamics._ops.cell_align as module

    calls = []
    monkeypatch.setattr(module, "_align_cell_op", lambda *args: calls.append(args))
    for backend in (None, "warp"):
        r, h = tensors()
        align_cell(r, h, backend=backend)
    assert len(calls) == 2


def test_legacy_bridge_writes_back_strided_positions(monkeypatch):
    """UBF-010 bridge regression; a stub is NOT Warp numerical evidence."""
    warp = ModuleType("warp")
    warp.from_torch = lambda tensor, dtype: tensor
    warp.device_from_torch = lambda device: device
    warp.int32 = torch.int32
    utils = ModuleType("nvalchemiops.dynamics.utils")

    def rotate(positions, cell, transform, **kwargs):
        positions.add_(7)
        cell.mul_(2)

    utils.align_cell = rotate
    bridge = ModuleType("nvalchemi.dynamics._ops._bridge")
    bridge._vec_type = bridge._mat_type = lambda dtype: dtype
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    r, h = tensors()
    r = torch.stack((r, r), -1)[..., 0]
    h = torch.stack((h, h), -1)[..., 0]
    before_r, before_h = r.clone(), h.clone()
    align_cell(r, h)
    assert torch.equal(r, before_r + 7) and torch.equal(h, before_h * 2)


def test_second_executor_and_all_declared_entrypoints(monkeypatch):
    """A second registered module uses the unchanged dispatcher."""
    module = ModuleType("cell_tools_second_executor")
    calls = []

    def align(positions, cell, transform, batch_idx):
        calls.append(batch_idx)
        positions.add_(3)
        return positions, cell

    module.align_cell = align
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import nvalchemiops.backend as backend_module

    registry = ImplementationRegistry(DEFAULT_IMPLEMENTATION_REGISTRY.implementations())
    registry.register(
        Implementation(
            implementation_id="test.cell-tools-second-v1",
            operation="cell_tools",
            family="test",
            executor=module.__name__,
            entrypoints=("align_cell",),
            executor_owner="ops",
            features=frozenset({"row_cell"}),
            max_gradient_order=0,
            default_strategy=True,
        )
    )
    monkeypatch.setattr(backend_module, "DEFAULT_IMPLEMENTATION_REGISTRY", registry)
    clear_entrypoint_cache()
    try:
        r, h = tensors()
        before = r.clone()
        align_cell(r, h, backend="test")
        assert torch.equal(r, before + 3) and len(calls) == 1
    finally:
        clear_entrypoint_cache()


def test_executor_pack_abi_and_unmodified_inputs():
    selected = resolve_backend(
        "torch_reference",
        operation="cell_tools",
        device="cpu",
        dtype=torch.float64,
        features={"six_cell_dofs"},
    )
    r, h = tensors()
    before_r, before_h = r.clone(), h.clone()
    idx = torch.zeros(2, dtype=torch.int32)
    ptr, ext_ptr = torch.tensor([0, 2]), torch.empty(2, dtype=torch.int64)

    def unavailable(*a, **k):
        pytest.fail("reference called legacy")

    def call(name, *a, **k):
        return execute_selected(selected, name, unavailable, *a, **k)

    call("extend_atom_ptr", ptr, ext_ptr)
    call("extend_batch_idx", idx, 2, 1, torch.empty(4, dtype=idx.dtype))
    ext = torch.empty(4, 3, dtype=r.dtype)
    out_r, out_h = torch.empty_like(r), torch.empty_like(h)
    for pack, unpack in [
        ("pack_positions_with_cell", "unpack_positions_with_cell"),
        ("pack_velocities_with_cell", "unpack_velocities_with_cell"),
    ]:
        assert call(pack, r, h, ext, ptr, ext_ptr, batch_idx=idx) is ext
        call(
            unpack, ext, out_r, out_h, atom_ptr=ptr, ext_atom_ptr=ext_ptr, batch_idx=idx
        )
        assert torch.equal(out_r, r) and torch.equal(out_h, torch.tril(h))
    call("pack_forces_with_cell", r, h, ext, ptr, ext_ptr, batch_idx=idx)
    call(
        "pack_masses_with_cell",
        torch.ones(2, dtype=r.dtype),
        torch.ones(1, dtype=r.dtype),
        torch.empty(4, dtype=r.dtype),
        ptr,
        ext_ptr,
    )
    assert torch.equal(r, before_r) and torch.equal(h, before_h)
