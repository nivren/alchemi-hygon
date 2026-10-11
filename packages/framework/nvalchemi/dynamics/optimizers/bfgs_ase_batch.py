# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Private ID-keyed BFGS lifecycle adapter for the existing Batch scheduler."""

from collections.abc import Mapping

import torch
from nvalchemiops.ase_unitcell_filter import ASEUnitCellFilter
from nvalchemiops.bfgs_ase import (
    ASEBFGSState,
    bfgs_ase_batch_step,
    bfgs_ase_prepare_state,
)

from nvalchemi._backend import resolve_compute_backend
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage


class _BFGSBatch:
    """Own only optimizer-specific per-ID state; reuse BaseDynamics scheduling."""

    def __init__(self, owner):
        self.owner = owner
        self.entries = {}
        self.batch = None
        self.ids = []
        self.ptr = []
        self.selection = None
        self.last_steps = {}
        self.variable = hasattr(owner, "_filter_options")
        self.restart_ids = None

    def ensure(self, batch):
        """Validate membership/identity before numerical work and reconcile by ID."""
        q = batch.positions
        ids = getattr(batch, "system_id", None)
        if (
            ids is None
            or ids.shape not in ((batch.num_graphs,), (batch.num_graphs, 1))
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.device != q.device
        ):
            raise ValueError(
                "BFGS Batch requires unique integer system_id on its device"
            )
        identifiers = ids.reshape(-1).tolist()
        if self.restart_ids is not None and set(identifiers) != self.restart_ids:
            raise ValueError("BFGS Batch restart requires the paired system IDs")
        ptr = batch.batch_ptr.tolist()
        if (
            len(set(identifiers)) != len(identifiers)
            or len(ptr) != len(identifiers) + 1
            or ptr[0] != 0
            or ptr[-1] != len(q)
            or any(b <= a for a, b in zip(ptr, ptr[1:]))
        ):
            raise ValueError(
                "BFGS Batch needs unique IDs and nonempty system boundaries"
            )
        shapes = {
            "positions": (batch.num_nodes, 3),
            "forces": (batch.num_nodes, 3),
            "energy": (batch.num_graphs, 1),
        }
        if self.variable:
            shapes.update(
                cell=(batch.num_graphs, 3, 3), stress=(batch.num_graphs, 3, 3)
            )
        slots = []
        for name, shape in shapes.items():
            v = getattr(batch, name, None)
            if (
                not isinstance(v, torch.Tensor)
                or v.shape != shape
                or v.dtype != torch.float64
                or v.device != q.device
                or v.requires_grad
                or v.layout != torch.strided
                or not v.is_contiguous()
                or not bool(torch.isfinite(v).all())
            ):
                raise ValueError(
                    f"BFGS Batch requires finite contiguous inference float64 {name} {shape}"
                )
            slots.append(v)
        if len({v.untyped_storage().data_ptr() for v in slots}) != len(slots):
            raise ValueError("BFGS Batch geometry/output slots must not alias")
        features = {"single_system", "batch", "heterogeneous"} | (
            {"variable_cell", "ase_unitcell"} if self.variable else {"fixed_cell"}
        )
        if self.selection is None:
            self.selection = resolve_compute_backend(
                self.owner.backend,
                operation="bfgs_ase_compat",
                device=q.device,
                dtype=q.dtype,
                features=features,
            )
        keys = (
            ("atomic_numbers", "pbc")
            if self.variable
            else ("atomic_numbers", "pbc", "cell")
        )
        candidates = {}
        for i, identifier in enumerate(identifiers):
            a, b = ptr[i : i + 2]
            signature = {
                k: (
                    getattr(batch, k)[a:b]
                    if k == "atomic_numbers"
                    else getattr(batch, k)[i : i + 1]
                )
                if getattr(batch, k, None) is not None
                else None
                for k in keys
            }
            entry = self.entries.get(identifier)
            if entry is None:
                adapter = None
                if self.variable:
                    options = dict(self.owner._filter_options)
                    ref = options["orig_cell"]
                    if ref is not None and not isinstance(ref, torch.Tensor):
                        options["orig_cell"] = torch.as_tensor(
                            ref, dtype=q.dtype, device=q.device
                        ).contiguous()
                    adapter = ASEUnitCellFilter.prepare(batch.cell[i], b - a, **options)
                state = bfgs_ase_prepare_state(
                    3 * (b - a) + (9 if self.variable else 0),
                    alpha=self.owner.alpha,
                    maxstep=self.owner.maxstep,
                    device=q.device,
                    system_id=identifier,
                    coordinate_mode="ase_unitcell" if self.variable else "cartesian",
                    selection=self.selection,
                )
                entry = dict(
                    state=state,
                    filter=adapter,
                    signature={
                        k: v.clone() if v is not None else None
                        for k, v in signature.items()
                    },
                    frozen=False,
                )
            else:
                state = entry["state"]
                state.validate()
                if (
                    state.device != q.device
                    or state.num_scalar_dofs
                    != 3 * (b - a) + (9 if self.variable else 0)
                    or state.system_id != identifier
                    or state.coordinate_mode
                    != ("ase_unitcell" if self.variable else "cartesian")
                ):
                    raise ValueError(
                        "BFGS Batch state identity/dof/device/mode mismatch"
                    )
                for key, saved in entry["signature"].items():
                    current = signature[key]
                    if (saved is None) != (current is None) or (
                        saved is not None
                        and (
                            saved.device != current.device
                            or saved.dtype != current.dtype
                            or not torch.equal(saved, current)
                        )
                    ):
                        raise ValueError(
                            f"BFGS Batch identity changed for {identifier}: {key}"
                        )
                if self.variable:
                    entry["filter"].validate()
                    ASEUnitCellFilter._cell("cell", batch.cell[i], q.device)
                restart = entry.get("restart")
                if restart is not None and (
                    not torch.equal(q[a:b], restart[0])
                    or (
                        self.variable
                        and not torch.equal(batch.cell[i : i + 1], restart[1])
                    )
                ):
                    raise ValueError(
                        "BFGS Batch restart differs from paired physical geometry"
                    )
            candidates[identifier] = entry
            state = entry["state"]
            slots.extend(
                (state.hessian, state.previous_positions, state.previous_forces)
            )
            if entry["filter"] is not None:
                slots.extend((entry["filter"].orig_cell, entry["filter"].mask))
        if len({v.untyped_storage().data_ptr() for v in slots}) != len(slots):
            raise ValueError("BFGS Batch geometry/state/Filter storage must not alias")
        for entry in candidates.values():
            entry.pop("restart", None)
        self.entries = candidates
        self.restart_ids = None
        self.ids, self.ptr, self.batch = identifiers, ptr, batch
        self.owner._bound_batch = batch
        self.last_steps = {i: v for i, v in self.last_steps.items() if i in candidates}

    def converged(self):
        """Return the strict ASE criterion independently for each system."""
        batch = self.batch
        results = []
        for i, identifier in enumerate(self.ids):
            a, b = self.ptr[i : i + 2]
            force = batch.forces[a:b]
            if self.variable:
                force, _ = self.entries[identifier]["filter"].generalized_forces(
                    force, batch.stress[i], batch.cell[i]
                )
            results.append(force.square().sum(-1).amax() < self.owner.fmax**2)
        return torch.stack(results)

    def update(self, batch, mask, *, prime):
        """Prepare scratch candidates, batch the Ops call, then commit geometry."""
        self.ensure(batch)
        if (
            mask.shape != (batch.num_graphs,)
            or mask.dtype != torch.bool
            or mask.device != batch.positions.device
        ):
            raise ValueError("BFGS Batch mask must be bool [B] on its device")
        if prime:
            self.owner._prime(batch)
        converged = self.owner._check_convergence(batch)
        done = torch.zeros_like(mask)
        if converged is not None:
            done[converged] = True
        active = []
        for i, eligible in enumerate(mask.tolist()):
            entry = self.entries[self.ids[i]]
            if eligible and bool(done[i]):
                entry["frozen"] = True
            if eligible and not entry["frozen"]:
                active.append(i)
        if not active:
            return
        qs, fs, states = [], [], []
        for i in active:
            a, b = self.ptr[i : i + 2]
            entry = self.entries[self.ids[i]]
            adapter = entry["filter"]
            qs.append(
                adapter.pack(batch.positions[a:b], batch.cell[i])
                if adapter is not None
                else batch.positions[a:b].clone()
            )
            fs.append(
                adapter.generalized_forces(
                    batch.forces[a:b], batch.stress[i], batch.cell[i]
                )[0]
                if adapter is not None
                else batch.forces[a:b].clone()
            )
            states.append(ASEBFGSState.from_state_dict(entry["state"].state_dict()))
        with torch.no_grad():
            results = bfgs_ase_batch_step(qs, fs, states, selection=self.selection)
            geometry = [
                self.entries[self.ids[i]]["filter"].unpack(q)
                if self.variable
                else (q, None)
                for i, q in zip(active, qs, strict=True)
            ]
            for i, candidate, (q, h), result in zip(
                active, states, geometry, results, strict=True
            ):
                a, b = self.ptr[i : i + 2]
                batch.positions[a:b].copy_(q)
                if h is not None:
                    batch.cell[i].copy_(h)
                state = self.entries[self.ids[i]]["state"]
                state.hessian.copy_(candidate.hessian)
                state.previous_positions.copy_(candidate.previous_positions)
                state.previous_forces.copy_(candidate.previous_forces)
                state.initialized, state.step_count = (
                    candidate.initialized,
                    candidate.step_count,
                )
                self.last_steps[self.ids[i]] = result
        self.owner._force_stamp = None

    def step(self, batch):
        """Use the existing hook loop after one shared initial force evaluation."""
        self.ensure(batch)
        self.owner._prime(batch)
        converged = self.owner._check_convergence(batch)
        if converged is not None and converged.numel() == batch.num_graphs:
            for entry in self.entries.values():
                entry["frozen"] = True
            self.owner._last_converged = converged
            self.owner._call_hooks(DynamicsStage.ON_CONVERGE, batch)
            return batch, converged
        return BaseDynamics.step(self.owner, batch)

    def snapshot(self):
        """Save per-ID legacy snapshots with separately versioned batch metadata."""
        if self.batch is None:
            raise RuntimeError("BFGS Batch must be bound before saving")
        result = {}
        for i, identifier in enumerate(self.ids):
            entry = self.entries[identifier]
            a, b = self.ptr[i : i + 2]
            snapshot = dict(
                version=1,
                step_count=self.owner.step_count,
                fmax=self.owner.fmax,
                custom_convergence=self.owner._custom_convergence,
                bfgs_state=entry["state"].state_dict(),
                positions=self.batch.positions[a:b].clone(),
                signature={
                    k: v.clone() if v is not None else None
                    for k, v in entry["signature"].items()
                },
            )
            if self.variable:
                snapshot.update(
                    kind="ase-unitcell-bfgs",
                    filter_state=entry["filter"].state_dict(),
                    cell=self.batch.cell[i : i + 1].clone(),
                )
            result[identifier] = dict(snapshot=snapshot, frozen=entry["frozen"])
        output = dict(
            version=1,
            kind="ase-bfgs-batch-unitcell"
            if self.variable
            else "ase-bfgs-batch-cartesian",
            entries=result,
        )
        if self.variable:
            options = dict(self.owner._filter_options)
            for name in ("mask", "orig_cell"):
                value = options[name]
                if value is not None:
                    options[name] = (
                        torch.as_tensor(
                            value,
                            dtype=torch.float64,
                            device=self.batch.positions.device,
                        )
                        .clone()
                        .contiguous()
                    )
            output["filter_options"] = options
        return output

    def restore(self, data, device=None):
        """Reuse the strict single-system snapshot validators before any mutation."""
        kind = (
            "ase-bfgs-batch-unitcell" if self.variable else "ase-bfgs-batch-cartesian"
        )
        if (
            not isinstance(data, Mapping)
            or set(data)
            != (
                {"version", "kind", "entries", "filter_options"}
                if self.variable
                else {"version", "kind", "entries"}
            )
            or type(data["version"]) is not int
            or data["version"] != 1
            or data["kind"] != kind
            or not isinstance(data["entries"], Mapping)
            or not data["entries"]
        ):
            raise ValueError("unsupported BFGS Batch restart schema/kind")
        entries = {}
        settings = None
        devices = set()
        source_devices = set()
        for identifier, item in data["entries"].items():
            if (
                type(identifier) is not int
                or not isinstance(item, Mapping)
                or set(item) != {"snapshot", "frozen"}
                or type(item["frozen"]) is not bool
            ):
                raise ValueError("invalid BFGS Batch restart ID/entry")
            # Only snapshot validation uses a temporary single-system wrapper.
            # No model evaluation or per-system dynamics loop is executed.
            temp = type(self.owner)(
                self.owner.model,
                backend=self.owner.backend,
                convergence_hook=self.owner.convergence_hook
                if self.owner._custom_convergence
                else None,
            )
            temp.load_state_dict(item["snapshot"], device=device)
            source_devices.add(item["snapshot"]["bfgs_state"]["hessian"].device)
            state = temp.bfgs_state
            if state.system_id != identifier:
                raise ValueError("BFGS Batch restart ID differs from state")
            current = (temp.step_count, temp.fmax, temp.alpha, temp.maxstep)
            if settings is not None and settings != current:
                raise ValueError(
                    "BFGS Batch restart requires matching optimizer settings"
                )
            settings = current
            devices.add(state.device)
            entries[identifier] = dict(
                state=state,
                filter=temp.filter_state if self.variable else None,
                signature=temp._signature,
                frozen=item["frozen"],
                restart=(temp._restart_positions, getattr(temp, "_restart_cell", None)),
            )
        if len(devices) != 1 or len(source_devices) != 1:
            raise ValueError("BFGS Batch restart requires one device")
        options = None
        if self.variable:
            options = data["filter_options"]
            if not isinstance(options, Mapping) or set(options) != set(
                self.owner._filter_options
            ):
                raise ValueError("invalid BFGS Batch restart admission Filter options")
            options = dict(options)
            target = next(iter(devices))
            for name in ("mask", "orig_cell"):
                value = options[name]
                if value is not None:
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.dtype != torch.float64
                        or value.requires_grad
                        or value.device != next(iter(source_devices))
                    ):
                        raise ValueError(
                            "invalid BFGS Batch restart Filter parameter tensor"
                        )
                    options[name] = value.to(device=target, copy=True)
            for entry in entries.values():
                saved = entry["filter"]
                prepared = ASEUnitCellFilter.prepare(
                    saved.orig_cell, saved.num_atoms, **options
                )
                if (
                    prepared.cell_factor != saved.cell_factor
                    or prepared.scalar_pressure != saved.scalar_pressure
                    or prepared.hydrostatic_strain != saved.hydrostatic_strain
                    or prepared.constant_volume != saved.constant_volume
                    or not torch.equal(prepared.mask, saved.mask)
                    or (
                        options["orig_cell"] is not None
                        and not torch.equal(prepared.orig_cell, saved.orig_cell)
                    )
                ):
                    raise ValueError(
                        "BFGS Batch restart Filter options differ from entries"
                    )
        self.entries = entries
        self.restart_ids = set(entries)
        if options is not None:
            self.owner._filter_options = options
        self.owner.step_count, self.owner.fmax, self.owner.alpha, self.owner.maxstep = (
            settings
        )
        self.owner._force_stamp = None


class _BFGSSharedForceHook:
    """Remember shared FusedStage evaluations without a second model call."""

    stage = DynamicsStage.AFTER_COMPUTE
    frequency = 1

    def __init__(self, owner, stage=DynamicsStage.AFTER_COMPUTE):
        self.owner = owner
        self.stage = stage

    def __call__(self, ctx, stage):
        owner = self.owner
        owner._ensure_state_initialized(ctx.batch)
        if stage == DynamicsStage.BEFORE_PRE_UPDATE and owner._force_stamp is not None:
            return
        owner._force_stamp = owner._stamp(ctx.batch)
        owner._checkpoint_positions = ctx.batch.positions.clone()
        if hasattr(owner, "_filter_options"):
            owner._checkpoint_cell = ctx.batch.cell.clone()
