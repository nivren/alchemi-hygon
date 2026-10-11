# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Device-local ASE 3.29 UnitCellFilter chart; no ASE runtime dependency."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch

from nvalchemiops.bfgs_ase import _tensor

__all__ = ["ASEUnitCellFilter"]


@dataclass(frozen=True)
class ASEUnitCellFilter:
    """One system's independent reference cell and ASE projection parameters.

    Use :meth:`prepare` to clone parameters on the geometry's device. Tensor
    operations are forward-only FP64 on CPU or Torch CUDA/HIP, current stream.
    Full 3x3 cell coordinates have ASE's row/transposition conventions.
    """

    orig_cell: torch.Tensor
    mask: torch.Tensor
    num_atoms: int
    cell_factor: float
    hydrostatic_strain: bool = False
    constant_volume: bool = False
    scalar_pressure: float = 0.0

    @staticmethod
    def _cell(name: str, value: torch.Tensor, device: torch.device) -> None:
        """Reject singular/nonfinite geometry without regularizing its spectrum."""
        _tensor(name, value, (3, 3), device)
        if bool(torch.linalg.det(value) == 0):
            raise ValueError(f"{name} is singular")

    def validate(self) -> None:
        """Check parameters and reference geometry, including loaded snapshots."""
        if type(self.num_atoms) is not int or self.num_atoms <= 0:
            raise ValueError("UnitCellFilter requires a positive integer num_atoms")
        if not isinstance(self.orig_cell, torch.Tensor):
            raise TypeError("orig_cell must be a Tensor")
        if self.orig_cell.device.type not in {"cpu", "cuda"}:
            raise ValueError("UnitCellFilter supports CPU or Torch CUDA/HIP")
        self._cell("orig_cell", self.orig_cell, self.orig_cell.device)
        _tensor("mask", self.mask, (3, 3), self.orig_cell.device)
        if (
            self.mask.untyped_storage().data_ptr()
            == self.orig_cell.untyped_storage().data_ptr()
        ):
            raise ValueError("UnitCellFilter reference cell and mask must not alias")
        if (
            isinstance(self.cell_factor, bool)
            or not isinstance(self.cell_factor, (int, float))
            or not math.isfinite(self.cell_factor)
            or self.cell_factor == 0
        ):
            raise ValueError("cell_factor must be a finite nonzero real scalar")
        if (
            type(self.hydrostatic_strain) is not bool
            or type(self.constant_volume) is not bool
        ):
            raise TypeError("UnitCellFilter projection flags must be bool")
        if (
            isinstance(self.scalar_pressure, bool)
            or not isinstance(self.scalar_pressure, (int, float))
            or not math.isfinite(self.scalar_pressure)
        ):
            raise ValueError("scalar_pressure must be a finite real scalar")

    @classmethod
    def prepare(
        cls,
        cell: torch.Tensor,
        num_atoms: int,
        *,
        mask: Any = None,
        cell_factor: float | None = None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        orig_cell: torch.Tensor | None = None,
        scalar_pressure: float = 0.0,
    ) -> ASEUnitCellFilter:
        """Clone ASE defaults and parameters; cell is a non-singular [3,3] tensor.

        Python/array masks are converted as FP64 parameter data on this device.
        Tensor masks may be bool/numeric, but must already be on this device and
        forward-only. Numerical geometry is never cast or implicitly transferred.
        """
        if not isinstance(cell, torch.Tensor):
            raise TypeError("cell must be a Tensor")
        cls._cell("cell", cell, cell.device)
        ref = cell if orig_cell is None else orig_cell
        cls._cell("orig_cell", ref, cell.device)
        if isinstance(mask, torch.Tensor) and (
            mask.device != cell.device or mask.requires_grad
        ):
            raise ValueError("mask tensor must be forward-only on the geometry device")
        m = (
            torch.ones(3, 3, dtype=cell.dtype, device=cell.device)
            if mask is None
            else torch.as_tensor(mask, dtype=cell.dtype, device=cell.device).clone()
        )
        if m.shape == (6,):
            m = m[torch.tensor([[0, 5, 4], [5, 1, 3], [4, 3, 2]], device=cell.device)]
        elif m.shape != (3, 3):
            raise ValueError("shape of mask should be (3,3) or (6,)")
        instance = cls(
            ref.clone(),
            m.contiguous(),
            num_atoms,
            float(num_atoms) if cell_factor is None else cell_factor,
            hydrostatic_strain,
            constant_volume,
            scalar_pressure,
        )
        instance.validate()
        return instance

    def _deformation(self, cell: torch.Tensor) -> torch.Tensor:
        """Return F=solve(H0,H).T without cell alignment."""
        self.validate()
        self._cell("cell", cell, self.orig_cell.device)
        return torch.linalg.solve(self.orig_cell, cell).T

    def pack(self, positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Return fresh [N+3,3] generalized coordinates, leaving geometry intact."""
        f = self._deformation(cell)
        _tensor("positions", positions, (self.num_atoms, 3), self.orig_cell.device)
        return torch.cat((torch.linalg.solve(f, positions.T).T, self.cell_factor * f))

    def unpack(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return validated candidate positions/cell; never mutate inputs."""
        self.validate()
        _tensor(
            "coordinates", coordinates, (self.num_atoms + 3, 3), self.orig_cell.device
        )
        eye = torch.eye(3, dtype=coordinates.dtype, device=coordinates.device)
        transform = eye + (coordinates[-3:] / self.cell_factor - eye).T * self.mask
        cell = (self.orig_cell @ transform).contiguous()
        positions = (coordinates[:-3] @ transform).contiguous()
        self._cell("candidate cell", cell, self.orig_cell.device)
        _tensor(
            "candidate positions", positions, (self.num_atoms, 3), self.orig_cell.device
        )
        return positions, cell

    def generalized_forces(
        self, forces: torch.Tensor, stress: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return generalized force rows and ASE projected Voigt stress.

        Stress [3,3] is positive in tension, in eV/angstrom cubed. Off-diagonal
        stress is symmetrized as ASE's Atoms.get_stress conversion does.
        Projection order is hydrostatic, mask, constant-volume trace removal.
        """
        f = self._deformation(cell)
        _tensor("forces", forces, (self.num_atoms, 3), self.orig_cell.device)
        _tensor("stress", stress, (3, 3), self.orig_cell.device)
        volume = torch.linalg.det(cell).abs()
        eye = torch.eye(3, dtype=forces.dtype, device=forces.device)
        virial = -volume * ((stress + stress.T) / 2 + self.scalar_pressure * eye)
        virial = torch.linalg.solve(f, virial.T).T
        if self.hydrostatic_strain:
            virial = eye * virial.trace() / 3
        virial = virial * self.mask
        if self.constant_volume:
            virial = virial - eye * virial.trace() / 3
        projected = (
            -torch.stack(
                (
                    virial[0, 0],
                    virial[1, 1],
                    virial[2, 2],
                    (virial[1, 2] + virial[2, 1]) / 2,
                    (virial[0, 2] + virial[2, 0]) / 2,
                    (virial[0, 1] + virial[1, 0]) / 2,
                )
            )
            / volume
        )
        generalized = torch.cat((forces @ f, virial / self.cell_factor))
        _tensor(
            "generalized forces",
            generalized,
            (self.num_atoms + 3, 3),
            self.orig_cell.device,
        )
        return generalized, projected

    def enthalpy(self, energy: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Return E+pV without replacing the physical model energy."""
        if not isinstance(energy, torch.Tensor):
            raise TypeError("energy must be a Tensor")
        self._deformation(cell)
        _tensor("energy", energy, tuple(energy.shape), self.orig_cell.device)
        return energy + self.scalar_pressure * torch.linalg.det(cell).abs()

    def state_dict(self) -> dict[str, Any]:
        """Clone versioned Filter parameters and reference cell on this device."""
        self.validate()
        return {
            "version": 1,
            **{
                f.name: getattr(self, f.name).clone()
                if isinstance(getattr(self, f.name), torch.Tensor)
                else getattr(self, f.name)
                for f in fields(self)
            },
        }

    @classmethod
    def from_state_dict(
        cls, data: Mapping[str, Any], *, device: torch.device | str | None = None
    ) -> ASEUnitCellFilter:
        """Validate before cloning, with optional explicit device-only migration."""
        names = {f.name for f in fields(cls)}
        if (
            not isinstance(data, Mapping)
            or set(data) != names | {"version"}
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise ValueError("unsupported UnitCellFilter snapshot schema")
        candidate = cls(**{k: data[k] for k in names})
        candidate.validate()
        target = candidate.orig_cell.device if device is None else torch.device(device)
        restored = cls(
            **{
                k: data[k].to(device=target, copy=True)
                if isinstance(data[k], torch.Tensor)
                else data[k]
                for k in names
            }
        )
        restored.validate()
        return restored
