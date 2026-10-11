# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""ASE UnitCellFilter variable-cell adapter for the shared dense BFGS Ops."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from nvalchemiops.ase_unitcell_filter import ASEUnitCellFilter
from nvalchemiops.bfgs_ase import ASEBFGSState, bfgs_ase_prepare_state, bfgs_ase_step

from nvalchemi._backend import resolve_compute_backend
from nvalchemi.data import Batch
from nvalchemi.dynamics.optimizers.bfgs_ase import ASECompatibleBFGS
from nvalchemi.dynamics.optimizers.bfgs_ase_batch import _BFGSBatch

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["ASECompatibleBFGSVariableCell"]


class ASECompatibleBFGSVariableCell(ASECompatibleBFGS):
    """FP64 BFGS with ASE 3.29 UnitCellFilter coordinates and ID-keyed Batch.

    Parameters
    ----------
    model : BaseModelMixin
        Model providing physical energy, forces and ASE tensile stress.
    mask : array-like or Tensor, optional
        ASE [xx,yy,zz,yz,xz,xy] mask or full [3,3]. Default all ones.
    cell_factor : float or None, optional
        Generalized cell-coordinate scale; default number of atoms.
    hydrostatic_strain, constant_volume : bool, optional
        ASE virial projections, in that order around masking. Constant volume
        is a trace projection, not an exact volume constraint. Default False.
    orig_cell : array-like or Tensor or None, optional
        Independent reference row cell; default first bound Batch cell.
    scalar_pressure : float, optional
        External pressure in eV/angstrom cubed; default zero.
    **kwargs : Any
        Fixed-cell BFGS options (alpha, maxstep, fmax, hooks, backend, etc.).
        maxstep and strict fmax apply to all N+3 generalized rows, not separate
        Cartesian force/stress thresholds. Batch potential energy stays physical;
        :meth:`get_enthalpy` returns E+pV. Filter state is independent per ID;
        default cell_factor is each system's atom count. No Atoms constraints.
    """

    __needs_keys__ = {"forces", "stress"}
    __provides_keys__ = {"positions", "cell"}

    def __init__(
        self,
        model: BaseModelMixin,
        *,
        mask: Any = None,
        cell_factor: float | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        orig_cell: Any = None,
        scalar_pressure: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self._filter_options = dict(
            mask=mask,
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            orig_cell=orig_cell,
            scalar_pressure=scalar_pressure,
        )
        self._filter_state: ASEUnitCellFilter | None = None
        self._checkpoint_cell: torch.Tensor | None = None
        self._bound_batch: Batch | None = None
        self.projected_stress: torch.Tensor | None = None

    @property
    def filter_state(self) -> ASEUnitCellFilter:
        """Return independently owned reference cell and projection parameters."""
        if self._batch_data is not None:
            if len(self._batch_data.entries) != 1:
                raise RuntimeError("use filter_states for multiple systems")
            return next(iter(self._batch_data.entries.values()))["filter"]
        if self._filter_state is None:
            raise RuntimeError("UnitCellFilter has not been initialized")
        return self._filter_state

    @property
    def filter_states(self) -> dict[int | None, ASEUnitCellFilter]:
        """Return per-ID ASE reference cells and projection parameters."""
        if self._batch_data is not None:
            return {
                key: entry["filter"] for key, entry in self._batch_data.entries.items()
            }
        return {self.bfgs_state.system_id: self.filter_state}

    def _generalized_forces(self, batch: Batch) -> torch.Tensor:
        """Map fresh physical forces/stress to all N+3 ASE rows."""
        force, self.projected_stress = self.filter_state.generalized_forces(
            batch.forces, batch.stress[0], batch.cell[0]
        )
        return force

    def _force_converged(self, forces: torch.Tensor) -> torch.Tensor:
        """Use the strict ASE joint generalized-force criterion."""
        if self._batch_data is not None:
            return self._batch_data.converged()
        if self._bound_batch is None:
            raise RuntimeError("UnitCellFilter has no bound Batch")
        return super()._force_converged(self._generalized_forces(self._bound_batch))

    def _stamp(self, batch: Batch) -> tuple[int, ...]:
        """Invalidate force cache for cell and stress mutations as well."""
        return super()._stamp(batch) + (
            batch.cell.data_ptr(),
            batch.cell._version,
            batch.stress.data_ptr(),
            batch.stress._version,
        )

    def _ensure_state_initialized(self, batch: Batch) -> None:
        """Bind one full-rank variable cell without aligning or triangularizing."""
        if self._use_batch(batch):
            self._batch_data.ensure(batch)
            return
        if (
            not isinstance(batch, Batch)
            or batch.num_graphs != 1
            or batch.num_nodes == 0
        ):
            raise ValueError("UnitCell BFGS M3 requires one nonempty system")
        q = batch.positions
        if q.shape != (batch.num_nodes, 3):
            raise ValueError("UnitCell BFGS positions must be [N,3]")
        specs = {
            "positions": q.shape,
            "forces": q.shape,
            "energy": (1, 1),
            "cell": (1, 3, 3),
            "stress": (1, 3, 3),
        }
        tensors = []
        for name, shape in specs.items():
            value = getattr(batch, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != torch.float64
                or value.device != q.device
                or value.requires_grad
                or value.layout != torch.strided
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"UnitCell BFGS requires a matching contiguous inference float64 {name} {shape} slot"
                )
            tensors.append(value)
        if len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors):
            raise ValueError("UnitCell BFGS geometry/output slots must not alias")
        ASEUnitCellFilter._cell("cell", batch.cell[0], q.device)
        if not bool(torch.isfinite(q).all()):
            raise ValueError("UnitCell BFGS positions must be finite")
        signature = {k: getattr(batch, k, None) for k in ("atomic_numbers", "pbc")}
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
                    raise ValueError(f"UnitCell BFGS identity changed: {k}")
        state = self._bfgs_state
        if state is not None:
            if (
                state.coordinate_mode != "ase_unitcell"
                or state.num_scalar_dofs != q.numel() + 9
                or state.dtype != q.dtype
                or state.device != q.device
                or state.system_id != self._system_id(batch)
            ):
                raise ValueError(
                    "UnitCell BFGS state identity/dof/dtype/device/mode mismatch"
                )
            self.filter_state.validate()
            if (
                self.filter_state.num_atoms != q.shape[0]
                or self.filter_state.orig_cell.device != q.device
            ):
                raise ValueError("UnitCellFilter state does not match the system")
        else:
            options = dict(self._filter_options)
            ref = options["orig_cell"]
            if ref is not None and not isinstance(ref, torch.Tensor):
                options["orig_cell"] = torch.as_tensor(
                    ref, dtype=q.dtype, device=q.device
                ).contiguous()
            adapter = ASEUnitCellFilter.prepare(batch.cell[0], q.shape[0], **options)
            selection = resolve_compute_backend(
                self.backend,
                operation="bfgs_ase_compat",
                device=q.device,
                dtype=q.dtype,
                features={"single_system", "variable_cell", "ase_unitcell"},
            )
            state = bfgs_ase_prepare_state(
                q.numel() + 9,
                alpha=self.alpha,
                maxstep=self.maxstep,
                device=q.device,
                dtype=q.dtype,
                system_id=self._system_id(batch),
                coordinate_mode="ase_unitcell",
                selection=selection,
            )
            self._bfgs_state, self._filter_state = state, adapter
            self._backend_selection = selection
            self._signature = {
                k: v.clone() if v is not None else None for k, v in signature.items()
            }
        if getattr(self, "_restart_positions", None) is not None:
            if not torch.equal(q, self._restart_positions) or not torch.equal(
                batch.cell, self._restart_cell
            ):
                raise ValueError(
                    "UnitCell BFGS restart differs from paired Batch positions/cell"
                )
            self._restart_positions = self._restart_cell = None
        if not hasattr(self, "_backend_selection"):
            self._backend_selection = resolve_compute_backend(
                self.backend,
                operation="bfgs_ase_compat",
                device=q.device,
                dtype=q.dtype,
                features={"single_system", "variable_cell", "ase_unitcell"},
            )
        tensors += [
            state.hessian,
            state.previous_positions,
            state.previous_forces,
            self.filter_state.orig_cell,
            self.filter_state.mask,
        ]
        if len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors):
            raise ValueError("UnitCell BFGS geometry, state and Filter must not alias")
        self._bound_batch = batch

    def compute(self, batch: Batch) -> Any:
        """Use existing model/Hooks dispatch and remember paired cell geometry."""
        result = super().compute(batch)
        self._checkpoint_cell = batch.cell.clone()
        return result

    def pre_update(self, batch: Batch) -> None:
        """Compute in scratch state and commit only after valid cell unpacking."""
        if self._use_batch(batch):
            return super().pre_update(batch)
        self._ensure_state_initialized(batch)
        if not self._active(batch):
            return
        self._prime(batch)
        if self._check_convergence(batch) is not None:
            return
        with torch.no_grad():
            q = self.filter_state.pack(batch.positions, batch.cell[0])
            force = self._generalized_forces(batch)
            candidate = ASEBFGSState.from_state_dict(self.bfgs_state.state_dict())
            result = bfgs_ase_step(
                q,
                force,
                candidate,
                system_id=self._system_id(batch),
                selection=self._backend_selection,
            )
            positions, cell = self.filter_state.unpack(q)
            state = self.bfgs_state
            batch.positions.copy_(positions)
            batch.cell[0].copy_(cell)
            state.hessian.copy_(candidate.hessian)
            state.previous_positions.copy_(candidate.previous_positions)
            state.previous_forces.copy_(candidate.previous_forces)
            state.initialized, state.step_count = (
                candidate.initialized,
                candidate.step_count,
            )
            self.last_step = result
        self._force_stamp = None
        self._checkpoint_positions = batch.positions.clone()
        self._checkpoint_cell = batch.cell.clone()

    def get_enthalpy(self, batch: Batch) -> torch.Tensor:
        """Return physical E+pV; does not overwrite Batch potential energy."""
        self._ensure_state_initialized(batch)
        if self._batch_data is not None:
            return torch.stack(
                [
                    self.filter_states[identifier].enthalpy(
                        batch.energy[i], batch.cell[i]
                    )
                    for i, identifier in enumerate(self._batch_data.ids)
                ]
            )
        return self.filter_state.enthalpy(batch.energy, batch.cell[0])

    def state_dict(self) -> dict[str, Any]:
        """Clone Filter and generalized history with paired physical geometry."""
        if self._batch_data is not None:
            return self._batch_data.snapshot()
        if (
            self._checkpoint_positions is None
            or self._checkpoint_cell is None
            or self._signature is None
        ):
            raise RuntimeError(
                "UnitCell BFGS must be evaluated before saving a restart"
            )
        return dict(
            version=1,
            kind="ase-unitcell-bfgs",
            step_count=self.step_count,
            fmax=self.fmax,
            custom_convergence=self._custom_convergence,
            bfgs_state=self.bfgs_state.state_dict(),
            filter_state=self.filter_state.state_dict(),
            positions=self._checkpoint_positions.clone(),
            cell=self._checkpoint_cell.clone(),
            signature={
                k: v.clone() if v is not None else None
                for k, v in self._signature.items()
            },
        )

    def load_state_dict(
        self, data: Mapping[str, Any], *, device: str | torch.device | None = None
    ) -> None:
        """Validate the whole continuation snapshot before atomically replacing it."""
        if isinstance(data, Mapping) and str(data.get("kind", "")).startswith(
            "ase-bfgs-batch"
        ):
            adapter = _BFGSBatch(self)
            adapter.restore(data, device=device)
            self._batch_data = adapter
            return
        keys = {
            "version",
            "kind",
            "step_count",
            "fmax",
            "custom_convergence",
            "bfgs_state",
            "filter_state",
            "positions",
            "cell",
            "signature",
        }
        if (
            not isinstance(data, Mapping)
            or set(data) != keys
            or type(data["version"]) is not int
            or data["version"] != 1
            or data["kind"] != "ase-unitcell-bfgs"
        ):
            raise ValueError("unsupported UnitCell BFGS restart schema/kind")
        count, fmax = data["step_count"], data["fmax"]
        if (
            type(count) is not int
            or count < 0
            or isinstance(fmax, bool)
            or not isinstance(fmax, (int, float))
            or not math.isfinite(fmax)
            or fmax <= 0
        ):
            raise ValueError("invalid UnitCell BFGS restart count/fmax")
        if (
            type(data["custom_convergence"]) is not bool
            or data["custom_convergence"] != self._custom_convergence
        ):
            raise ValueError("UnitCell BFGS restart requires matching convergence mode")
        state = ASEBFGSState.from_state_dict(data["bfgs_state"], device=device)
        adapter = ASEUnitCellFilter.from_state_dict(data["filter_state"], device=device)
        if (
            state.coordinate_mode != "ase_unitcell"
            or state.num_scalar_dofs != 3 * adapter.num_atoms + 9
            or state.device != adapter.orig_cell.device
            or data["bfgs_state"]["hessian"].device
            != data["filter_state"]["orig_cell"].device
        ):
            raise ValueError("UnitCell BFGS restart mode/dof/Filter device mismatch")
        source_device = data["bfgs_state"]["hessian"].device
        shapes = {"positions": (adapter.num_atoms, 3), "cell": (1, 3, 3)}
        for name, shape in shapes.items():
            value = data[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != torch.float64
                or value.device != source_device
                or value.requires_grad
                or value.layout != torch.strided
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"invalid UnitCell BFGS restart {name}")
        ASEUnitCellFilter._cell("restart cell", data["cell"][0], source_device)
        signature = data["signature"]
        if not isinstance(signature, Mapping) or set(signature) != {
            "atomic_numbers",
            "pbc",
        }:
            raise ValueError("invalid UnitCell BFGS identity signature")
        for k, value in signature.items():
            if value is None and k == "pbc":
                continue
            shape = (adapter.num_atoms,) if k == "atomic_numbers" else (1, 3)
            dtypes = (
                (torch.int32, torch.int64) if k == "atomic_numbers" else (torch.bool,)
            )
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype not in dtypes
                or value.device != source_device
                or value.requires_grad
            ):
                raise ValueError(f"invalid UnitCell BFGS identity signature {k}")
        copied = {
            k: v.to(device=state.device, copy=True) if v is not None else None
            for k, v in signature.items()
        }
        q, h = (
            data[k].to(device=state.device, copy=True) for k in ("positions", "cell")
        )
        self._bfgs_state, self._filter_state, self._signature = state, adapter, copied
        self._restart_positions, self._restart_cell = q, h
        self._checkpoint_positions, self._checkpoint_cell = q.clone(), h.clone()
        self.alpha, self.maxstep, self.fmax = state.alpha, state.maxstep, float(fmax)
        self.step_count = count
        self.last_step = None
        self.projected_stress = None
        self._force_stamp = None
        self._bound_batch = None
        self._batch_data = None
        self.__dict__.pop("_backend_selection", None)
