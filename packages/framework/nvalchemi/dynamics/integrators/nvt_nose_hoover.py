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
"""
NVT integrator via Nosé-Hoover chain (NHC) thermostat.

The NHC thermostat is deterministic, time-reversible, and ergodic.
It uses Yoshida-Suzuki factorization with the Martyna-Tobias-Klein
(MTK) equations.

The step is split around the force evaluation:

* ``pre_update``:  NHC half → v half kick → r full step
* [model evaluates F(r(t+dt))]
* ``post_update``: v half kick → NHC half

Per-system state: ``dt``, ``temperature``, ``thermostat_time``,
chain positions ``nhc_eta [M, C]``, chain velocities ``nhc_eta_dot [M, C]``,
chain masses ``nhc_Q [M, C]``, and NHC scratch tensors
``nhc_ke2 [M]``, ``nhc_total_scale [M]``, ``nhc_step_scale [M]``,
``nhc_dt_chain [M]``.

References
----------
Martyna, Tobias, Klein (1994); Tuckerman et al. (2006).
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch
from nvalchemiops.backend import BackendSelection

from nvalchemi._backend import resolve_compute_backend
from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_state_batch, _to_per_system
from nvalchemi.dynamics._ops.nose_hoover import (
    nhc_chain_update,
    nhc_compute_masses,
    nhc_position_update,
    nhc_velocity_half_step,
)
from nvalchemi.dynamics._units import fs_to_internal_time
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NVTNoseHoover"]


class NVTNoseHoover(BaseDynamics):
    r"""NVT integrator via Nosé-Hoover chain (NHC) thermostat.

    Deterministic extended-Lagrangian thermostat that rigorously samples
    the canonical ensemble for ergodic systems.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    dt : float or torch.Tensor
        Integration timestep in femtoseconds ``[M]`` or scalar.
    temperature : float or torch.Tensor
        Target temperature in Kelvin ``[M]`` or scalar.
    thermostat_time : float or torch.Tensor
        Thermostat coupling time :math:`\tau_T` in femtoseconds ``[M]`` or scalar.
        Controls how tightly the thermostat couples to the system;
        typical values are 10–100 :math:`\times \Delta t`.
    chain_length : int, optional
        Number of links in the Nosé-Hoover chain.  Default 3.
    yoshida_order : int, optional
        Order of the Yoshida-Suzuki integrator.  Must be 3 or 5.
        Default 3.
    backend : str or None, optional
        None retains Warp; torch_reference selects the fixed-cell Torch path.
        Per-system vectors support ordinary Batch; sampler refill requires scalar
        constructor parameters until arrival parameter mapping is defined.
    n_steps : int, optional
        Total steps for :meth:`run`.
    hooks : list[Hook], optional
        Initial hooks.
    convergence_hook : ConvergenceHook or dict, optional
        Convergence criterion.
    **kwargs
        Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.

    Attributes
    ----------
    __needs_keys__ : set[str]
        ``{"forces"}``.
    __provides_keys__ : set[str]
        ``{"positions", "velocities"}``.
    """

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"positions", "velocities"}

    # Domain-parallel intent (read by the dynamics coordinator; inert
    # single-process). This integrator couples its thermostat to the mesh-global
    # kinetic energy + DOF; its NHC controller state is replicated and kept
    # byte-identical across ranks.
    __dd_thermo_kind__: str = "nhc"
    __dd_replicated__: tuple[str, ...] = ("nhc_eta", "nhc_eta_dot")

    def __init__(
        self,
        model: BaseModelMixin,
        dt: float | torch.Tensor,
        temperature: float | torch.Tensor,
        thermostat_time: float | torch.Tensor,
        chain_length: int = 3,
        yoshida_order: Literal[3, 5] = 3,
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        self._dt_init = fs_to_internal_time(dt)
        self._temperature_init = temperature
        self._thermostat_time_init = fs_to_internal_time(thermostat_time)
        if (
            isinstance(chain_length, bool)
            or not isinstance(chain_length, int)
            or not 1 <= chain_length <= 8
        ):
            raise ValueError("chain_length must be an integer in [1,8]")
        if (
            isinstance(yoshida_order, bool)
            or not isinstance(yoshida_order, int)
            or yoshida_order not in (3, 5)
        ):
            raise ValueError("yoshida_order must be 3 or 5")
        self.backend = backend
        self._backend_selection: BackendSelection | None = None
        self.chain_length = chain_length
        self.yoshida_order = yoshida_order

    def _init_state(self, batch: Batch) -> None:
        self._backend_selection = resolve_compute_backend(
            self.backend,
            operation="nose_hoover",
            device=batch.positions.device,
            dtype=batch.positions.dtype,
            gradient_order=0,
            features={"fixed_cell"},
        )
        self._state = self._build_state(batch, batch.num_graphs, 0)
        pending = getattr(self, "_pending_restart_state", None)
        if pending is not None:
            try:
                self._restore_state(pending, batch)
            except (ValueError, TypeError):
                # A failed pending restore must not leave a runnable fresh chain.
                del self._state
                raise
            del self._pending_restart_state

    def _build_state(self, batch: Batch, n: int, first_graph: int) -> Batch:
        dev, dtype = batch.device, batch.positions.dtype
        parameters = {}
        for name, value in (
            ("dt", self._dt_init),
            ("temperature", self._temperature_init * KB_EV),
            ("thermostat_time", self._thermostat_time_init),
        ):
            if first_graph or n != batch.num_graphs:
                # Per-system constructor vectors have no identity for future arrivals.
                # Do not silently reuse graph 0's parameters during inflight refill.
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    raise ValueError(
                        "NHC refill requires scalar constructor parameters; per-system arrival parameters need an explicit mapping"
                    )
            tensor = _to_per_system(value, n, dev, dtype)
            if tensor.shape != (n,):
                raise ValueError(f"{name} must be scalar or have shape [M]")
            if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
                raise ValueError(f"{name} must be positive and finite")
            parameters[name] = tensor
        # Upstream fix NHC-REFILL: use the appended systems' actual atom counts,
        # not dummy_masses[:n] or the mean size of remaining + new systems.
        # BaseDynamics appends arrivals at the tail of the updated Batch.
        counts = (
            batch.batch_ptr[first_graph + 1 : first_graph + n + 1]
            - batch.batch_ptr[first_graph : first_graph + n]
        )
        if bool((counts <= 0).any()):
            raise ValueError("NHC systems must have at least one atom")
        start = int(batch.batch_ptr[first_graph])
        end = int(batch.batch_ptr[first_graph + n])
        idx = batch.batch_idx[start:end].int() - first_graph
        Q = nhc_compute_masses(
            parameters["temperature"],
            parameters["thermostat_time"],
            batch.atomic_masses[start:end],
            idx,
            self.chain_length,
            selection=self._backend_selection,
        )
        return _make_state_batch(
            {
                **parameters,
                "nhc_ndof": (counts * 3).to(dtype=dtype),
                "nhc_eta": torch.zeros(n, self.chain_length, device=dev, dtype=dtype),
                "nhc_eta_dot": torch.zeros(
                    n, self.chain_length, device=dev, dtype=dtype
                ),
                "nhc_Q": Q,
                **{
                    key: torch.zeros(n, device=dev, dtype=dtype)
                    for key in (
                        "nhc_ke2",
                        "nhc_total_scale",
                        "nhc_step_scale",
                        "nhc_dt_chain",
                    )
                },
            },
            dev,
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        if not 0 <= n <= template_batch.num_graphs:
            raise ValueError("new system count exceeds updated Batch")
        # Also reject vectors when all old systems have exited (first_graph=0).
        for value in (
            self._dt_init,
            self._temperature_init,
            self._thermostat_time_init,
        ):
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                raise ValueError(
                    "NHC refill requires scalar constructor parameters; per-system arrival parameters need an explicit mapping"
                )
        return self._build_state(template_batch, n, template_batch.num_graphs - n)

    _restart_keys = (
        "dt",
        "temperature",
        "thermostat_time",
        "nhc_ndof",
        "nhc_eta",
        "nhc_eta_dot",
        "nhc_Q",
    )

    def state_dict(self) -> dict[str, Any]:
        """Integrator continuation only; Batch/model/hooks remain caller-owned."""
        result = {
            "version": 1,
            "step_count": self.step_count,
            "chain_length": self.chain_length,
            "yoshida_order": self.yoshida_order,
        }
        if hasattr(self, "_state"):
            result["integrator_state"] = {
                key: getattr(self._state, key).detach().clone()
                for key in self._restart_keys
            }
        elif hasattr(self, "_pending_restart_state"):
            result["integrator_state"] = {
                key: value.clone() for key, value in self._pending_restart_state.items()
            }
        return result

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported NHC restart version")
        if (
            state.get("chain_length") != self.chain_length
            or state.get("yoshida_order") != self.yoshida_order
        ):
            raise ValueError("NHC restart chain_length/yoshida_order mismatch")
        try:
            steps = operator.index(state["step_count"])
        except (KeyError, TypeError) as exc:
            raise ValueError("NHC restart step_count must be an integer") from exc
        if isinstance(state["step_count"], bool) or steps < 0:
            raise ValueError("NHC restart step_count must be non-negative")
        tensors = state.get("integrator_state")
        copied = None
        if tensors is not None:
            if not isinstance(tensors, Mapping) or set(tensors) != set(
                self._restart_keys
            ):
                raise ValueError("NHC restart integrator_state keys mismatch")
            copied = {}
            for key in self._restart_keys:
                value = tensors[key]
                if not isinstance(value, torch.Tensor) or value.dtype not in (
                    torch.float32,
                    torch.float64,
                ):
                    raise TypeError("NHC restart state must contain FP32/FP64 tensors")
                copied[key] = value.detach().clone()
            if copied["dt"].ndim != 1:
                raise ValueError("NHC restart dt must have shape [M]")
            M = copied["dt"].shape[0]
            for key, value in copied.items():
                shape = (
                    (M, self.chain_length)
                    if key in ("nhc_eta", "nhc_eta_dot", "nhc_Q")
                    else (M,)
                )
                if value.shape != shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"NHC restart {key} has invalid shape/values")
                if key not in ("nhc_eta", "nhc_eta_dot") and not bool(
                    (value > 0).all()
                ):
                    raise ValueError(f"NHC restart {key} must be positive")
            if hasattr(self, "_state") and M != self._state.num_graphs:
                raise ValueError("NHC restart system count mismatch")
            if hasattr(self, "_state") and not torch.equal(
                copied["nhc_ndof"].to(self._state.nhc_ndof), self._state.nhc_ndof
            ):
                raise ValueError("NHC restart atom counts mismatch")
        # Validate first, then commit state so a rejected restore changes nothing.
        if copied is None:
            self.__dict__.pop("_state", None)
            self.__dict__.pop("_pending_restart_state", None)
        elif hasattr(self, "_state"):
            for key, value in copied.items():
                target = getattr(self._state, key)
                target.copy_(value.to(device=target.device, dtype=target.dtype))
        else:
            self._pending_restart_state = copied
        self.step_count = steps

    def _restore_state(self, copied, batch):
        counts = batch.batch_ptr[1:] - batch.batch_ptr[:-1]
        if copied["dt"].shape != (batch.num_graphs,):
            raise ValueError("NHC restart system count mismatch")
        if not torch.equal(copied["nhc_ndof"].to(counts.device), counts * 3):
            raise ValueError("NHC restart atom counts mismatch")
        for key, value in copied.items():
            getattr(self._state, key).copy_(
                value.to(device=batch.device, dtype=batch.positions.dtype)
            )

    def _preserve_inactive_chain(self, batch, mask, update):
        self._ensure_state_initialized(batch)
        keys = (
            "nhc_eta",
            "nhc_eta_dot",
            "nhc_ke2",
            "nhc_total_scale",
            "nhc_step_scale",
            "nhc_dt_chain",
        )
        saved = {key: getattr(self._state, key)[~mask].clone() for key in keys}
        try:
            return update(batch, mask)
        finally:
            # Upstream fix NHC-MASK: BaseDynamics restores Batch atom fields,
            # but not the private thermostat state. Inactive FusedStage graphs
            # must not advance their chain while another stage owns them.
            for key, value in saved.items():
                getattr(self._state, key)[~mask] = value

    def step(self, batch):
        # Restore private state inside pre/post update, before AFTER_* hooks.
        # FusedStage supplies its own mask, independently of exit_status.
        if hasattr(batch, "status") and batch.status is not None:
            self._nhc_step_mask = (
                batch.status.reshape(-1)[: batch.num_graphs] < self.exit_status
            )
        try:
            return super().step(batch)
        finally:
            self.__dict__.pop("_nhc_step_mask", None)

    def _masked_pre_update(self, batch, mask):
        self._preserve_inactive_chain(batch, mask, super()._masked_pre_update)

    def _masked_post_update(self, batch, mask):
        self._preserve_inactive_chain(batch, mask, super()._masked_post_update)

    def masked_update(self, batch, mask):
        self._preserve_inactive_chain(batch, mask, super().masked_update)

    def pre_update(self, batch: Batch) -> None:
        mask = getattr(self, "_nhc_step_mask", None)
        if mask is None:
            return self._pre_update_impl(batch)
        return self._preserve_inactive_chain(
            batch, mask, lambda b, _: self._pre_update_impl(b)
        )

    def _pre_update_impl(self, batch: Batch) -> None:
        """NHC half-step → velocity half-kick → position full-step.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *velocities* updated in-place.
        """
        nhc_chain_update(
            batch.velocities,
            batch.atomic_masses,
            self._state.nhc_eta,
            self._state.nhc_eta_dot,
            self._state.nhc_Q,
            self._state.temperature,
            self._state.dt,
            self._state.nhc_ndof,
            self._state.nhc_ke2,
            self._state.nhc_total_scale,
            self._state.nhc_step_scale,
            self._state.nhc_dt_chain,
            batch.batch_idx.int(),
            # Upstream fix NHC-YOSHIDA: forward the documented 3/5 selection.
            nloops=self.yoshida_order,
            selection=self._backend_selection,
        )
        nhc_velocity_half_step(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch_idx.int(),
            selection=self._backend_selection,
        )
        nhc_position_update(
            batch.positions,
            batch.velocities,
            self._state.dt,
            batch.batch_idx.int(),
            selection=self._backend_selection,
        )

    def post_update(self, batch: Batch) -> None:
        mask = getattr(self, "_nhc_step_mask", None)
        if mask is None:
            return self._post_update_impl(batch)
        return self._preserve_inactive_chain(
            batch, mask, lambda b, _: self._post_update_impl(b)
        )

    def _post_update_impl(self, batch: Batch) -> None:
        """Velocity half-kick → NHC half-step (symmetric closure).

        Parameters
        ----------
        batch : Batch
            Current batch; *velocities* updated in-place.
        """
        nhc_velocity_half_step(
            batch.velocities,
            batch.forces,
            batch.atomic_masses,
            self._state.dt,
            batch.batch_idx.int(),
            selection=self._backend_selection,
        )
        nhc_chain_update(
            batch.velocities,
            batch.atomic_masses,
            self._state.nhc_eta,
            self._state.nhc_eta_dot,
            self._state.nhc_Q,
            self._state.temperature,
            self._state.dt,
            self._state.nhc_ndof,
            self._state.nhc_ke2,
            self._state.nhc_total_scale,
            self._state.nhc_step_scale,
            self._state.nhc_dt_chain,
            batch.batch_idx.int(),
            # Upstream fix NHC-YOSHIDA: forward the documented 3/5 selection.
            nloops=self.yoshida_order,
            selection=self._backend_selection,
        )
