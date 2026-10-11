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

"""Local ASE 3.29-compatible BFGS state and device-neutral Ops dispatch.

This is a new alchemi-hygon operation, not an NVIDIA BFGS port. ASE is used
only as a test oracle. See docs/TORCH_ASE_BFGS_CONTRACT.md and
docs/TORCH_ASE_UNITCELL_FILTER_CONTRACT.md for the per-system FP64 contracts,
and docs/TORCH_ASE_BFGS_BATCH_CONTRACT.md for grouped batched updates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

from nvalchemiops._backend_types import LEGACY_IMPLEMENTATION_ID
from nvalchemiops.backend import (
    BackendSelection,
    BackendUnavailableError,
    resolve_backend,
)
from nvalchemiops.executor import execute_selected

__all__ = [
    "ASEBFGSState",
    "ASEBFGSStep",
    "BFGSNumericalError",
    "bfgs_ase_prepare_state",
    "bfgs_ase_step",
    "bfgs_ase_batch_step",
]

_OPERATION = "bfgs_ase_compat"
_FEATURES = frozenset({"single_system", "fixed_cell"})
_ALGORITHM = "ase-bfgs-3.29.0"
_SCHEMA = 1


def _features(coordinate_mode: str) -> frozenset[str]:
    """Identify Cartesian or ASE UnitCellFilter numerical coordinates."""
    if coordinate_mode == "cartesian":
        return _FEATURES
    if coordinate_mode == "ase_unitcell":
        return frozenset({"single_system", "variable_cell", "ase_unitcell"})
    raise ValueError("unsupported BFGS coordinate_mode")


class BFGSNumericalError(RuntimeError):
    """The ASE step cannot produce a finite result without changing its algorithm."""


def _positive_parameter(name: str, value: float) -> None:
    """Reject invalid hyperparameters without coercion or precision changes."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _system_id(value: int | None) -> None:
    """Validate optional identity metadata separately from tensor indices."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError("system_id must be an integer or None")


def _dofs(value: int) -> None:
    """Validate single-system generalized coordinates arranged as vec3 rows."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("num_scalar_dofs must be an integer")
    if value <= 0 or value % 3:
        raise ValueError("num_scalar_dofs must be positive and divisible by 3")


def _tensor(
    name: str, value: torch.Tensor, shape: tuple[int, ...], device: torch.device
) -> None:
    """Check supported layout, device, dtype, gradients and finite values."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if (
        value.shape != shape
        or value.layout != torch.strided
        or not value.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous strided with shape {shape}")
    if value.dtype != torch.float64 or value.device != device:
        raise ValueError(f"{name} must have dtype float64 on {device}")
    if value.requires_grad:
        raise ValueError(f"{name}: BFGS supports forward-only inputs")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _no_alias(*tensors: torch.Tensor) -> None:
    """Reject shared storage, including disjoint views, before in-place updates."""
    pointers = [tensor.untyped_storage().data_ptr() for tensor in tensors]
    if len(set(pointers)) != len(pointers):
        raise ValueError(
            "BFGS positions, forces and state tensors must not alias storage"
        )


@dataclass
class ASEBFGSState:
    """One independent dense Hessian and pre-step coordinate/force history.

    Parameters
    ----------
    hessian : torch.Tensor
        Dense [d,d] FP64 energy Hessian.
    previous_positions, previous_forces : torch.Tensor
        Independent [d/3,3] tensors, populated after the first step.
    alpha, maxstep : float
        Positive initial curvature and global maximum row displacement.
    system_id : int or None
        Optional stable identity; never a Batch slot index managed by this Ops.
    initialized, step_count : bool, int
        Host lifecycle metadata. Model evaluation and stopping are caller-owned.
    coordinate_mode : str
        Cartesian (3N) or ASE UnitCellFilter (3N+9) coordinate/force contract.
    """

    hessian: torch.Tensor
    previous_positions: torch.Tensor
    previous_forces: torch.Tensor
    alpha: float = 70.0
    maxstep: float = 0.2
    system_id: int | None = None
    initialized: bool = False
    step_count: int = 0
    coordinate_mode: str = "cartesian"

    @property
    def num_scalar_dofs(self) -> int:
        """Return coordinate width from the Hessian."""
        return self.hessian.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the unchanged tensor precision."""
        return self.hessian.dtype

    @property
    def device(self) -> torch.device:
        """Return the actual state tensor device."""
        return self.hessian.device

    def validate(self) -> None:
        """Validate all tensors and lifecycle metadata before use or restoration."""
        _positive_parameter("alpha", self.alpha)
        _positive_parameter("maxstep", self.maxstep)
        _system_id(self.system_id)
        if not isinstance(self.initialized, bool):
            raise TypeError("initialized must be a bool")
        if (
            isinstance(self.step_count, bool)
            or not isinstance(self.step_count, int)
            or self.step_count < 0
        ):
            raise ValueError("step_count must be a nonnegative integer")
        if self.initialized != (self.step_count > 0):
            raise ValueError("initialized and step_count disagree")
        if not isinstance(self.hessian, torch.Tensor) or self.hessian.ndim != 2:
            raise ValueError("hessian must be a square Tensor")
        d = self.num_scalar_dofs
        _dofs(d)
        _features(self.coordinate_mode)
        if self.coordinate_mode == "ase_unitcell" and d < 12:
            raise ValueError("UnitCell BFGS needs at least one atom and 9 cell dofs")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("BFGS state supports only CPU or Torch CUDA/HIP devices")
        _tensor("hessian", self.hessian, (d, d), self.device)
        _tensor("previous_positions", self.previous_positions, (d // 3, 3), self.device)
        _tensor("previous_forces", self.previous_forces, (d // 3, 3), self.device)
        _no_alias(self.hessian, self.previous_positions, self.previous_forces)

    def state_dict(self) -> dict[str, Any]:
        """Return a versioned snapshot with independent device-local tensor storage."""
        self.validate()
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        if self.coordinate_mode == "cartesian":
            values.pop("coordinate_mode")
        return {
            "schema_version": _SCHEMA if self.coordinate_mode == "cartesian" else 2,
            "algorithm": _ALGORITHM,
            "num_scalar_dofs": self.num_scalar_dofs,
            **{
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in values.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls, data: Mapping[str, Any], *, device: torch.device | str | None = None
    ) -> ASEBFGSState:
        """Validate and clone a snapshot; only an explicit device request moves data.

        Parameters
        ----------
        data : Mapping
            Snapshot produced by state_dict, possibly loaded with torch.load.
        device : torch.device or str, optional
            Explicit restoration target. No implicit device or dtype conversion.
        """
        if not isinstance(data, Mapping):
            raise ValueError("BFGS snapshot must be a mapping")
        version = data.get("schema_version")
        names = {field.name for field in fields(cls)}
        if type(version) is int and version == 1:
            names.remove("coordinate_mode")
        expected = names | {
            "schema_version",
            "algorithm",
            "num_scalar_dofs",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("BFGS snapshot keys do not match schema")
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] not in (1, 2)
            or (
                data["schema_version"] == 2
                and data.get("coordinate_mode") != "ase_unitcell"
            )
            or data["algorithm"] != _ALGORITHM
        ):
            raise ValueError("unsupported BFGS snapshot schema or algorithm")
        _dofs(data["num_scalar_dofs"])
        values = {name: data[name] for name in names}
        candidate = cls(**values)
        candidate.validate()
        if candidate.num_scalar_dofs != data["num_scalar_dofs"]:
            raise ValueError("snapshot num_scalar_dofs does not match tensors")
        target = candidate.device if device is None else torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("restoration target must be CPU or Torch CUDA/HIP")
        restored = cls(
            **{
                name: value.to(device=target, copy=True)
                if isinstance(value, torch.Tensor)
                else value
                for name, value in values.items()
            }
        )
        restored.validate()
        return restored


@dataclass(frozen=True)
class ASEBFGSStep:
    """Independent device-local diagnostics for one successful BFGS step."""

    eigenvalues: torch.Tensor
    raw_step: torch.Tensor
    applied_step: torch.Tensor
    scale: torch.Tensor
    max_step_length: torch.Tensor


def _selection(
    backend: str | None,
    selection: BackendSelection | None,
    device: torch.device,
    dtype: torch.dtype,
    coordinate_mode: str = "cartesian",
    *,
    batch: bool = False,
) -> BackendSelection:
    """Validate both fresh and cached selection against actual input capability."""
    from nvalchemiops.backend import DEFAULT_IMPLEMENTATION_REGISTRY

    features = _features(coordinate_mode)
    if batch:
        features = (features - {"single_system"}) | {"batch", "heterogeneous"}
    if selection is None:
        selection = resolve_backend(
            backend,
            operation=_OPERATION,
            device=device,
            dtype=dtype,
            features=features,
        )
    if (
        selection.operation != _OPERATION
        or selection.device != device.type
        or selection.dtype != str(dtype).removeprefix("torch.")
        or selection.gradient_order != 0
        or not features.issubset(selection.features)
    ):
        raise BackendUnavailableError(
            "BFGS selection does not match operation/device/dtype/features/gradient contract"
        )
    impl = DEFAULT_IMPLEMENTATION_REGISTRY.get(selection.implementation_id)
    if selection.implementation_id == LEGACY_IMPLEMENTATION_ID:
        # Legacy has no capability tags; the local executor closure gives the
        # precise absent-upstream-BFGS error instead of a generic tag failure.
        return selection
    if impl is None or not impl.supports(
        device=device.type,
        dtype=selection.dtype,
        gradient_order=0,
        features=frozenset(selection.features),
    ):
        raise BackendUnavailableError(
            "selected BFGS implementation does not support the actual input contract"
        )
    return selection


def _legacy_unavailable(*args: Any, **kwargs: Any) -> Any:
    """Preserve legacy semantics while explicitly rejecting absent upstream BFGS."""
    raise BackendUnavailableError(
        "NVIDIA upstream has no legacy BFGS; request backend='torch_reference'"
    )


def bfgs_ase_prepare_state(
    num_scalar_dofs: int,
    *,
    alpha: float = 70.0,
    maxstep: float = 0.2,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    system_id: int | None = None,
    backend: str | None = "torch_reference",
    selection: BackendSelection | None = None,
    coordinate_mode: str = "cartesian",
) -> ASEBFGSState:
    """Allocate an independent single-system state through the registered executor.

    Parameters
    ----------
    num_scalar_dofs : int
        Positive generalized-coordinate width, divisible by three.
    alpha, maxstep : float
        Initial Hessian diagonal and uniform maximum atom displacement.
    dtype, device : torch.dtype, torch.device or str
        FP64 on CPU or Torch CUDA/HIP. No hidden precision/device conversion.
    system_id : int or None
        Optional system identity for subsequent consistency checks.
    backend : str or None
        Explicit implementation family, default torch_reference. Legacy is absent.
    selection : BackendSelection or None
        Optional pre-resolved decision, checked against the actual request.
    coordinate_mode : str
        cartesian (default) or ase_unitcell; changes capability and snapshot
        metadata, while retaining exactly the same numerical step.
    """
    _dofs(num_scalar_dofs)
    _features(coordinate_mode)
    if coordinate_mode == "ase_unitcell" and num_scalar_dofs < 12:
        raise ValueError("UnitCell BFGS needs at least one atom and 9 cell dofs")
    _positive_parameter("alpha", alpha)
    _positive_parameter("maxstep", maxstep)
    _system_id(system_id)
    if dtype != torch.float64:
        raise ValueError("ASE BFGS M1 supports only float64")
    target = torch.device(device)
    selected = _selection(backend, selection, target, dtype, coordinate_mode)
    return execute_selected(
        selected,
        "bfgs_ase_prepare_state",
        _legacy_unavailable,
        num_scalar_dofs,
        alpha=alpha,
        maxstep=maxstep,
        dtype=dtype,
        device=target,
        system_id=system_id,
        coordinate_mode=coordinate_mode,
    )


def bfgs_ase_step(
    positions: torch.Tensor,
    forces: torch.Tensor,
    state: ASEBFGSState,
    *,
    system_id: int | None = None,
    backend: str | None = "torch_reference",
    selection: BackendSelection | None = None,
) -> ASEBFGSStep:
    """Apply one ASE-compatible step, committing coordinates and history together.

    Parameters
    ----------
    positions, forces : torch.Tensor
        Contiguous [d/3,3] FP64 coordinates and negative energy gradients.
        UnitCellFilter uses N+3 rows; the caller owns geometry pack/unpack.
        Positions mutate; forces remain unchanged. No tensor storage may overlap.
    state : ASEBFGSState
        Independent state on the same device; mutated after a finite step exists.
    system_id : int or None
        If specified, must match state's stable identity.
    backend : str or None
        Explicit implementation family. Unsupported requests never fall back.
    selection : BackendSelection or None
        Optional cached implementation decision, checked against these tensors.

    Returns
    -------
    ASEBFGSStep
        Fresh spectrum, direction, applied displacement, scale and raw max length.
    """
    if not isinstance(state, ASEBFGSState):
        raise TypeError("state must be ASEBFGSState")
    state.validate()
    _system_id(system_id)
    if system_id is not None and system_id != state.system_id:
        raise ValueError("BFGS system_id does not match state")
    shape = (state.num_scalar_dofs // 3, 3)
    _tensor("positions", positions, shape, state.device)
    _tensor("forces", forces, shape, state.device)
    _no_alias(
        positions,
        forces,
        state.hessian,
        state.previous_positions,
        state.previous_forces,
    )
    selected = _selection(
        backend, selection, positions.device, positions.dtype, state.coordinate_mode
    )
    return execute_selected(
        selected, "bfgs_ase_step", _legacy_unavailable, positions, forces, state
    )


def bfgs_ase_batch_step(
    positions: Sequence[torch.Tensor],
    forces: Sequence[torch.Tensor],
    states: Sequence[ASEBFGSState],
    *,
    backend: str | None = "torch_reference",
    selection: BackendSelection | None = None,
) -> tuple[ASEBFGSStep, ...]:
    """Update independent systems with grouped batched math and atomic commit.

    All tensors are contiguous FP64 on one device, with one coordinate mode.
    No input/state storage may overlap. Multiple systems require unique IDs.
    Different DOFs are grouped without padding; all groups succeed before commit.
    See docs/TORCH_ASE_BFGS_BATCH_CONTRACT.md for lifecycle and error semantics.
    """
    if not states or len(positions) != len(states) or len(forces) != len(states):
        raise ValueError("BFGS batch requires nonempty equally sized sequences")
    if any(not isinstance(s, ASEBFGSState) for s in states):
        raise TypeError("BFGS batch states must be ASEBFGSState")
    first = states[0]
    tensors = []
    ids = [s.system_id for s in states]
    if len(states) > 1 and (None in ids or len(set(ids)) != len(ids)):
        raise ValueError("BFGS batch requires unique system IDs")
    for q, f, s in zip(positions, forces, states, strict=True):
        s.validate()
        if s.device != first.device or s.coordinate_mode != first.coordinate_mode:
            raise ValueError("BFGS batch requires one device and coordinate mode")
        shape = (s.num_scalar_dofs // 3, 3)
        _tensor("positions", q, shape, s.device)
        _tensor("forces", f, shape, s.device)
        tensors.extend((q, f, s.hessian, s.previous_positions, s.previous_forces))
    _no_alias(*tensors)
    selected = _selection(
        backend, selection, first.device, first.dtype, first.coordinate_mode, batch=True
    )
    return execute_selected(
        selected, "bfgs_ase_batch_step", _legacy_unavailable, positions, forces, states
    )
