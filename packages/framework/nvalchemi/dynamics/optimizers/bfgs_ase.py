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
"""Local ASE-compatible BFGS, delegating all numerical updates to Ops."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from nvalchemiops.bfgs_ase import (
    ASEBFGSState,
    ASEBFGSStep,
    bfgs_ase_prepare_state,
    bfgs_ase_step,
)

from nvalchemi._backend import resolve_compute_backend
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.optimizers.bfgs_ase_batch import (
    _BFGSBatch,
    _BFGSSharedForceHook,
)

if TYPE_CHECKING:
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["ASECompatibleBFGS"]


class ASECompatibleBFGS(BaseDynamics):
    """Dense, fixed-cell BFGS with ASE 3.29.0 numerical semantics.

    This is a local extension, absent from the reviewed NVIDIA baseline.
    Accepts nonempty FP64 systems in a Batch on CPU or CUDA/HIP. Multiple
    systems require unique integer system_id; independent state follows IDs
    through reorder/refill. Equal DOFs use grouped batched arithmetic/eigh.
    A full Hessian costs O((3N)^2) memory and its decomposition O((3N)^3).
    No damping, line search or eigenvalue clamp is applied.

    Parameters
    ----------
    model : BaseModelMixin
        Model returning forces; energy/force slots must exist in the Batch.
    alpha : float, optional
        Initial Hessian diagonal in eV/angstrom squared. Default 70.
    maxstep : float, optional
        Maximum atomic displacement in angstrom. Default 0.2.
    fmax : float, optional
        Strict maximum force norm stopping threshold in eV/angstrom.
        Default 0.05. A custom convergence_hook overrides this criterion.
    n_steps : int or None, optional
        Maximum updates requested by run().
    hooks : list[Hook] or None, optional
        Existing Dynamics hooks, including BEFORE_COMPUTE neighbors.
    convergence_hook : ConvergenceHook, dict or None, optional
        Custom stopping criterion; None selects the strict ASE criterion.
    backend : str or None, optional
        Explicit Ops backend, default torch_reference. There is no legacy
        NVIDIA BFGS implementation to select with None or warp.
    **kwargs : Any
        Existing BaseDynamics options, including native sampler/refill.
        FusedStage runs automatic inflight admission; standalone callers can
        use refill_check explicitly. Inter-rank stages are unsupported.
    """

    __needs_keys__ = {"forces"}
    __provides_keys__ = {"positions"}

    def __init__(
        self,
        model: BaseModelMixin,
        *,
        alpha: float = 70.0,
        maxstep: float = 0.2,
        fmax: float = 0.05,
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        backend: str | None = "torch_reference",
        **kwargs: Any,
    ) -> None:
        for name, value in (("alpha", alpha), ("maxstep", maxstep), ("fmax", fmax)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"BFGS {name} must be finite and positive")
        if any(
            kwargs.get(k, -1) not in (-1, None) for k in ("prior_rank", "next_rank")
        ):
            raise NotImplementedError("BFGS inter-rank stages are unsupported")
        self.alpha, self.maxstep, self.fmax = float(alpha), float(maxstep), float(fmax)
        self._custom_convergence = convergence_hook is not None
        if convergence_hook is None:
            convergence_hook = ConvergenceHook(
                criteria={
                    "key": "forces",
                    "threshold": self.fmax,
                    "custom_op": self._force_converged,
                }
            )
        super().__init__(
            model=model,
            hooks=hooks,
            convergence_hook=convergence_hook,
            n_steps=n_steps,
            **kwargs,
        )
        self.backend = backend
        self._bfgs_state: ASEBFGSState | None = None
        self.last_step: ASEBFGSStep | None = None
        self._force_stamp: tuple[int, ...] | None = None
        self._signature: dict[str, torch.Tensor | None] | None = None
        self._checkpoint_positions: torch.Tensor | None = None
        self._batch_data: _BFGSBatch | None = None
        self.register_hook(_BFGSSharedForceHook(self))
        self.register_hook(_BFGSSharedForceHook(self, DynamicsStage.BEFORE_PRE_UPDATE))

    def _use_batch(self, batch: Batch) -> bool:
        """Activate ID-keyed state once heterogeneous scheduling is required."""
        if self._batch_data is None and (
            batch.num_graphs != 1 or self.sampler is not None
        ):
            adapter = _BFGSBatch(self)
            state = self._bfgs_state
            if state is not None:
                if state.system_id is None:
                    raise ValueError(
                        "promoting BFGS history to Batch requires a system ID"
                    )
                adapter.entries[state.system_id] = dict(
                    state=state,
                    filter=getattr(self, "_filter_state", None),
                    signature=self._signature,
                    frozen=False,
                )
                paired = getattr(self, "_restart_positions", None)
                if paired is not None:
                    ids = getattr(batch, "system_id", None)
                    if ids is None or state.system_id not in ids.reshape(-1).tolist():
                        raise ValueError(
                            "promoting a BFGS restart requires its paired system ID"
                        )
                    adapter.entries[state.system_id]["restart"] = (
                        paired,
                        getattr(self, "_restart_cell", None),
                    )
            self._batch_data = adapter
        return self._batch_data is not None

    @property
    def bfgs_states(self) -> dict[int | None, ASEBFGSState]:
        """Return independent Ops states keyed by stable system identity."""
        if self._batch_data is not None:
            return {
                key: entry["state"] for key, entry in self._batch_data.entries.items()
            }
        state = self.bfgs_state
        return {state.system_id: state}

    @property
    def last_steps(self) -> dict[int | None, ASEBFGSStep]:
        """Return successful per-ID numerical diagnostics."""
        if self._batch_data is not None:
            return dict(self._batch_data.last_steps)
        return (
            {}
            if self.last_step is None
            else {self.bfgs_state.system_id: self.last_step}
        )

    @property
    def bfgs_state(self) -> ASEBFGSState:
        """Return initialized Ops state; raise before binding a Batch."""
        if self._batch_data is not None:
            if len(self._batch_data.entries) != 1:
                raise RuntimeError("use bfgs_states for multiple systems")
            return next(iter(self._batch_data.entries.values()))["state"]
        if self._bfgs_state is None:
            raise RuntimeError("BFGS state has not been initialized")
        return self._bfgs_state

    def _force_converged(self, forces: torch.Tensor) -> torch.Tensor:
        """Use the single-system strict ASE force norm criterion."""
        if self._batch_data is not None:
            return self._batch_data.converged()
        return (forces.square().sum(-1).max() < self.fmax**2).reshape(1)

    def _system_id(self, batch: Batch) -> int | None:
        """Read the one-system identifier without moving numerical state."""
        ids = getattr(batch, "system_id", None)
        if ids is None:
            return None
        if ids.numel() != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("single-system BFGS requires one integer system_id")
        return int(ids.item())

    def _stamp(self, batch: Batch) -> tuple[int, ...]:
        """Track ordinary external coordinate/force mutations between steps."""
        return (
            id(batch),
            batch.positions.data_ptr(),
            batch.positions._version,
            batch.forces.data_ptr(),
            batch.forces._version,
        )

    def _active(self, batch: Batch) -> bool:
        """Respect the existing exit-status boundary before any numerical update."""
        status = getattr(batch, "status", None)
        return status is None or bool((status < self.exit_status).all())

    def _ensure_state_initialized(self, batch: Batch) -> None:
        """Bind one fixed system and reject history/geometry mismatches."""
        if self._use_batch(batch):
            self._batch_data.ensure(batch)
            return
        if (
            not isinstance(batch, Batch)
            or batch.num_graphs != 1
            or batch.num_nodes == 0
        ):
            raise ValueError("BFGS M2 requires a single nonempty system in Batch")
        q, f = batch.positions, getattr(batch, "forces", None)
        if q.dtype != torch.float64:
            raise ValueError("BFGS positions must use float64")
        if (
            not isinstance(f, torch.Tensor)
            or f.shape != q.shape
            or f.dtype != q.dtype
            or f.device != q.device
        ):
            raise ValueError("BFGS requires matching float64 positions/forces slots")
        if (
            not q.is_contiguous()
            or not f.is_contiguous()
            or q.requires_grad
            or f.requires_grad
        ):
            raise ValueError(
                "BFGS coordinates/forces must be contiguous inference tensors"
            )
        energy = getattr(batch, "energy", None)
        if (
            not isinstance(energy, torch.Tensor)
            or energy.shape != (1, 1)
            or energy.dtype != q.dtype
            or energy.device != q.device
            or energy.requires_grad
        ):
            raise ValueError(
                "BFGS requires a matching float64 Dynamics energy [1,1] slot"
            )
        storages = [v.untyped_storage().data_ptr() for v in (q, f, energy)]
        if len(set(storages)) != 3:
            raise ValueError("BFGS positions/forces/energy must not share storage")
        signature = {
            k: getattr(batch, k, None) for k in ("atomic_numbers", "cell", "pbc")
        }
        if self._signature is not None:
            for k, saved in self._signature.items():
                current = signature[k]
                if (saved is None) != (current is None) or (
                    saved is not None
                    and (
                        saved.device != current.device
                        or saved.dtype != current.dtype
                        or not torch.equal(saved, current)
                    )
                ):
                    raise ValueError(
                        f"BFGS fixed geometry/atomic identity changed: {k}"
                    )
        state = self._bfgs_state
        if state is not None:
            if (
                state.coordinate_mode != "cartesian"
                or state.num_scalar_dofs != q.numel()
                or state.dtype != q.dtype
                or state.device != q.device
            ):
                raise ValueError(
                    "BFGS state dof/dtype/device does not match the system"
                )
            if state.system_id != self._system_id(batch):
                raise ValueError(
                    "BFGS system_id changed; initialize a separate optimizer"
                )
        else:
            self._backend_selection = resolve_compute_backend(
                self.backend,
                operation="bfgs_ase_compat",
                device=q.device,
                dtype=q.dtype,
                features={"single_system", "fixed_cell"},
            )
            self._bfgs_state = bfgs_ase_prepare_state(
                q.numel(),
                alpha=self.alpha,
                maxstep=self.maxstep,
                device=q.device,
                dtype=q.dtype,
                system_id=self._system_id(batch),
                selection=self._backend_selection,
            )
            self._signature = {
                k: v.clone() if v is not None else None for k, v in signature.items()
            }
        if getattr(self, "_restart_positions", None) is not None:
            if not torch.equal(q, self._restart_positions):
                raise ValueError(
                    "BFGS restart positions differ from the paired Batch snapshot"
                )
            self._restart_positions = None
        if not hasattr(self, "_backend_selection"):
            self._backend_selection = resolve_compute_backend(
                self.backend,
                operation="bfgs_ase_compat",
                device=q.device,
                dtype=q.dtype,
                features={"single_system", "fixed_cell"},
            )

    def _prime(self, batch: Batch) -> None:
        """Evaluate current forces through the existing compute hook boundary."""
        if self._force_stamp != self._stamp(batch):
            self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
            self.compute(batch)
            self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

    def compute(self, batch: Batch) -> Any:
        """Delegate to BaseDynamics and mark forces fresh before observer hooks."""
        frozen = []
        if self._batch_data is not None:
            for i, identifier in enumerate(self._batch_data.ids):
                if self._batch_data.entries[identifier]["frozen"]:
                    a, b = self._batch_data.ptr[i : i + 2]
                    fields = {
                        "forces": batch.forces[a:b].clone(),
                        "energy": batch.energy[i].clone(),
                    }
                    if self._batch_data.variable:
                        fields["stress"] = batch.stress[i].clone()
                    frozen.append((i, a, b, fields))
        outputs = super().compute(batch)
        for i, a, b, fields in frozen:
            batch.forces[a:b].copy_(fields.pop("forces"))
            for name, value in fields.items():
                getattr(batch, name)[i].copy_(value)
        self._force_stamp = self._stamp(batch)
        return outputs

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Prime/check the initial structure, then use the BaseDynamics step.

        Only this initial stopping boundary differs from the base loop. Hooks
        around actual updates, model evaluation and run open/close are inherited.
        """
        if self._use_batch(batch):
            return self._batch_data.step(batch)
        self._ensure_state_initialized(batch)
        if self._active(batch):
            self._prime(batch)
            converged = self._check_convergence(batch)
            if converged is not None:
                self._last_converged = converged
                self._checkpoint_positions = batch.positions.clone()
                self._call_hooks(DynamicsStage.ON_CONVERGE, batch)
                return batch, converged
        result = super().step(batch)
        self._checkpoint_positions = batch.positions.clone()
        return result

    def pre_update(self, batch: Batch) -> None:
        """Apply one Ops BFGS update using current model forces."""
        if self._use_batch(batch):
            status = getattr(batch, "status", None)
            mask = (
                torch.ones(
                    batch.num_graphs, dtype=torch.bool, device=batch.positions.device
                )
                if status is None
                else status.reshape(-1) < self.exit_status
            )
            self._batch_data.update(batch, mask, prime=True)
            return
        self._ensure_state_initialized(batch)
        if not self._active(batch):
            return
        self._prime(batch)
        if self._check_convergence(batch) is not None:
            return
        with torch.no_grad():
            self.last_step = bfgs_ase_step(
                batch.positions,
                batch.forces,
                self.bfgs_state,
                system_id=self._system_id(batch),
                selection=self._backend_selection,
            )
        self._force_stamp = None
        self._checkpoint_positions = batch.positions.clone()

    def post_update(self, batch: Batch) -> None:
        """No second numerical update; forces are evaluated at the new coordinates."""

    def _validate_mask(self, batch: Batch, mask: torch.Tensor) -> None:
        """Require a same-device, one-system boolean mask."""
        self._ensure_state_initialized(batch)
        if (
            mask.shape != (1,)
            or mask.dtype != torch.bool
            or mask.device != batch.positions.device
        ):
            raise ValueError("BFGS mask must be bool [1] on the Batch device")

    def _masked_pre_update(self, batch: Batch, mask: torch.Tensor) -> None:
        """Skip inactive systems before evaluating or mutating their dense history."""
        if self._use_batch(batch):
            self._batch_data.update(batch, mask, prime=False)
            return
        self._validate_mask(batch, mask)
        if bool(mask[0]):
            self.pre_update(batch)

    def _masked_post_update(self, batch: Batch, mask: torch.Tensor) -> None:
        """Preserve the masked post-update entrance without modifying history."""
        if self._use_batch(batch):
            self._batch_data.ensure(batch)
            if (
                mask.shape != (batch.num_graphs,)
                or mask.dtype != torch.bool
                or mask.device != batch.positions.device
            ):
                raise ValueError("BFGS Batch mask must be bool [B] on its device")
            return
        self._validate_mask(batch, mask)
        if bool(mask[0]):
            self.post_update(batch)

    def refill_check(self, batch: Batch, exit_status: int) -> Batch | None:
        """Reuse native sampler budgets, sinks and refill; prime new arrivals."""
        self._ensure_state_initialized(batch)
        result = BaseDynamics.refill_check(self, batch, exit_status)
        if result is not None and result is not batch:
            self._force_stamp = None
            self._prime(result)
        return result

    def _sync_state_to_batch(self, remaining_indices, n_new, template_batch):
        """Reconcile independent BFGS histories when native scheduling refills."""
        if self._use_batch(template_batch):
            if remaining_indices.numel() == 0 and n_new == 0:
                self._batch_data.entries.clear()
                self._batch_data.last_steps.clear()
                self._batch_data.ids = []
            else:
                self._batch_data.ensure(template_batch)
                self._batch_data.last_steps = {
                    i: v
                    for i, v in self._batch_data.last_steps.items()
                    if i in self._batch_data.entries
                }
            self._force_stamp = None
        else:
            super()._sync_state_to_batch(remaining_indices, n_new, template_batch)

    def state_dict(self) -> dict[str, Any]:
        """Clone continuation state; the paired Batch and model are saved separately."""
        if self._batch_data is not None:
            return self._batch_data.snapshot()
        if self._checkpoint_positions is None or self._signature is None:
            raise RuntimeError("BFGS must be bound/evaluated before saving a restart")
        return {
            "version": 1,
            "step_count": self.step_count,
            "fmax": self.fmax,
            "custom_convergence": self._custom_convergence,
            "bfgs_state": self.bfgs_state.state_dict(),
            "positions": self._checkpoint_positions.clone(),
            "signature": {
                k: v.clone() if v is not None else None
                for k, v in self._signature.items()
            },
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, device: str | torch.device | None = None
    ) -> None:
        """Validate and restore a local tensor continuation snapshot atomically.

        An explicit device moves the complete snapshot, without casting dtype.
        The next entrance checks the paired Batch and recomputes model forces.
        """
        if isinstance(state, Mapping) and str(state.get("kind", "")).startswith(
            "ase-bfgs-batch"
        ):
            adapter = _BFGSBatch(self)
            adapter.restore(state, device=device)
            self._batch_data = adapter
            return
        if not isinstance(state, Mapping):
            raise TypeError("BFGS restart must be a mapping")
        if set(state) != {
            "version",
            "step_count",
            "fmax",
            "custom_convergence",
            "bfgs_state",
            "positions",
            "signature",
        }:
            raise ValueError("BFGS restart keys do not match version 1")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported BFGS restart version")
        count = state["step_count"]
        if type(count) is not int or count < 0:
            raise ValueError("BFGS restart step_count must be nonnegative integer")
        fmax = state["fmax"]
        if (
            isinstance(fmax, bool)
            or not isinstance(fmax, (int, float))
            or not math.isfinite(fmax)
            or fmax <= 0
        ):
            raise ValueError("BFGS restart fmax must be finite and positive")
        restored = ASEBFGSState.from_state_dict(state["bfgs_state"], device=device)
        if restored.coordinate_mode != "cartesian":
            raise ValueError("fixed-cell BFGS restart requires Cartesian state")
        if (
            type(state["custom_convergence"]) is not bool
            or state["custom_convergence"] != self._custom_convergence
        ):
            raise ValueError(
                "BFGS restart requires the same default/custom convergence mode; recreate custom hooks separately"
            )
        q = state["positions"]
        if (
            not isinstance(q, torch.Tensor)
            or q.shape != restored.previous_positions.shape
            or q.dtype != restored.dtype
            or q.device != state["bfgs_state"]["hessian"].device
            or q.requires_grad
            or not torch.isfinite(q).all()
        ):
            raise ValueError("BFGS restart positions are invalid")
        signature = state["signature"]
        if not isinstance(signature, Mapping) or set(signature) != {
            "cell",
            "pbc",
            "atomic_numbers",
        }:
            raise ValueError("BFGS restart signature keys are invalid")
        shapes = {"atomic_numbers": (q.shape[0],), "cell": (1, 3, 3), "pbc": (1, 3)}
        dtypes = {
            "atomic_numbers": (torch.int32, torch.int64),
            "cell": (torch.float64,),
            "pbc": (torch.bool,),
        }
        for k, v in signature.items():
            if v is None and k != "atomic_numbers":
                continue
            if (
                not isinstance(v, torch.Tensor)
                or v.shape != shapes[k]
                or v.dtype not in dtypes[k]
                or v.device != q.device
                or v.requires_grad
                or not torch.isfinite(v).all()
            ):
                raise ValueError(f"BFGS restart identity signature is invalid: {k}")
        copied = {
            k: v.to(device=restored.device, copy=True) if v is not None else None
            for k, v in signature.items()
        }
        q_copy = q.to(device=restored.device, copy=True)
        self._bfgs_state, self._signature = restored, copied
        self._restart_positions = q_copy
        self._checkpoint_positions = q_copy.clone()
        self.alpha, self.maxstep, self.fmax = (
            restored.alpha,
            restored.maxstep,
            float(fmax),
        )
        self.step_count = count
        self.last_step = None
        self._force_stamp = None
        self._batch_data = None
        self.__dict__.pop("_backend_selection", None)
