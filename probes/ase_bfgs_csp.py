# SPDX-License-Identifier: Apache-2.0
"""Bounded real molecular-crystal FIRE2→ASE-BFGS Batch/inflight acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.filters import UnitCellFilter
from ase.io import read
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    FIRE2,
    ASECompatibleBFGS,
    ASECompatibleBFGSVariableCell,
    FusedStage,
)
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--backend", choices=("torch_reference", "hip", "hip.bfgs-ase-jacobi-streams-v1"), default="torch_reference"
    )
    parser.add_argument("--variable", action="store_true")
    parser.add_argument("--inflight", action="store_true")
    parser.add_argument("--check-steps", type=int, default=3)
    parser.add_argument("--qualification-only", action="store_true",
                        help="Bounded ASE/independent step validation; does not require or certify full convergence")
    parser.add_argument("--perturb-positions", type=float, default=0.0,
                        help="Diagnostic Gaussian position perturbation in Angstrom, deterministic seed 7300+i")
    parser.add_argument("--ase-shared-forces", action="store_true",
                        help="Qualification only: ASE CPU math uses workflow model force/stress inputs; no CPU MLIP parity claim")
    parser.add_argument(
        "--profile-bfgs-steps",
        type=int,
        default=0,
        help="Profile a bounded warmed FusedStage window; instrumented timing is not a benchmark",
    )
    parser.add_argument("--fire-steps", type=int, default=20)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/wangleping/.cache/mace/mace-mpa-0-medium.model"),
    )
    parser.add_argument(
        "--cifs",
        nargs="+",
        type=Path,
        default=[
            Path("/data/csp_data/perf_46/formal_c1_1_100_z1_46.cif"),
            Path("/data/csp_data/perf_v2_sorted/perf_v2_92/formal_c1_2_1000_z1_46.cif"),
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Explicit paired Batch/BFGS continuation after the FIRE stage; no sampler cursor restoration",
    )
    parser.add_argument(
        "--capture-eigh",
        type=Path,
        help="Explicit diagnostic CPU capture of the last matrix per DOF; excluded from timing acceptance",
    )
    args = parser.parse_args()
    if (
        not args.cifs
        or args.fmax <= 0
        or args.max_steps < 1
        or args.fire_steps < 1
        or args.check_steps < 0
        or args.profile_bfgs_steps < 0
        or args.max_wall_seconds <= 0
        or args.perturb_positions < 0
        or not np.isfinite(args.perturb_positions)
    ):
        raise ValueError("probe requires CIF inputs and positive execution bounds")
    if args.qualification_only and (args.inflight or args.check_steps < 1 or args.resume is not None):
        raise ValueError("bounded qualification requires fresh static inputs and at least one checked update")
    if args.ase_shared_forces and not args.qualification_only:
        raise ValueError("shared-force ASE oracle is limited to explicit bounded qualification")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if args.capture_eigh is not None:
        import nvalchemiops.bfgs_ase_hip as hip_executor

        args.capture_eigh.mkdir(parents=True, exist_ok=True)
        original_eigh = hip_executor._eigh

        def capture_eigh(h, **kwargs):
            torch.save(h.cpu(), args.capture_eigh / f"last-{h.shape[-1]}.pt")
            return original_eigh(h, **kwargs)

        hip_executor._eigh = capture_eigh
    structures = [read(path) for path in args.cifs]
    assert all(len(a) > 0 for a in structures)
    assert all(a.pbc.all() for a in structures)
    for i, atoms in enumerate(structures):
        if args.perturb_positions:
            atoms.positions += np.random.default_rng(7300+i).normal(
                scale=args.perturb_positions, size=atoms.positions.shape)
    model = MACEWrapper.from_checkpoint(
        args.checkpoint,
        device=device,
        dtype=torch.float64,
        enable_cueq=False,
        compile_model=False,
    )
    model.set_config(
        "active_outputs",
        {"energy", "forces", "stress"} if args.variable else {"energy", "forces"},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    for i, atoms in enumerate(structures * (2 if args.inflight else 1)):
        data = AtomicData.from_atoms(atoms, dtype=torch.float64).to(device)
        data.forces = torch.zeros_like(data.positions)
        data.energy = torch.zeros(1, 1, dtype=torch.float64, device=device)
        data.add_node_property("velocities", torch.zeros_like(data.positions))
        if args.variable:
            data.stress = torch.zeros(1, 3, 3, dtype=torch.float64, device=device)
        data.add_system_property("source_index", torch.tensor([[i]], device=device))
        data.add_system_property("system_id", torch.tensor([[i]], device=device))
        rows.append(data)

    class Dataset:
        def __len__(self):
            return len(rows)

        def get_metadata(self, i):
            return rows[i].num_nodes, 0

        def __getitem__(self, i):
            return rows[i].clone(), {}

    def neighbor():
        return NeighborListHook(
            model.model_config.neighbor_config,
            skin=0,
            backend="torch_reference",
            stage=DynamicsStage.BEFORE_COMPUTE,
        )

    cls = ASECompatibleBFGSVariableCell if args.variable else ASECompatibleBFGS
    bfgs = cls(model, backend=args.backend, fmax=args.fmax, device_type=args.device)
    fire = FIRE2(
        model=model,
        dt=0.01,
        n_steps=args.fire_steps,
        backend="torch_reference",
        device_type=args.device,
    )
    sink = HostMemory(capacity=len(rows))
    kwargs = dict(device_type=args.device)
    if args.inflight:
        kwargs.update(
            sampler=SizeAwareSampler(Dataset(), max_atoms=sum(map(len, structures)),
                                     max_batch_size=len(structures)),
            sinks=[sink],
            refill_frequency=1,
        )
    stage = FusedStage([(0, fire), (1, bfgs)], **kwargs)
    stage.register_hook(neighbor())
    resume = None
    if args.resume is not None:
        if args.inflight:
            raise ValueError("probe resume does not restore a sampler cursor")
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if bool((resume["batch"].status < 1).any()):
            raise ValueError(
                "probe continuation requires every system to have entered BFGS"
            )
        bfgs.load_state_dict(resume["bfgs"], device=device)
        stage.step_count = resume["stage_step_count"]
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    times = {"model": [], "bfgs_update": []}
    workflow = [True]
    started = [0.0]

    def model_pre(module, inputs):
        if workflow[0]:
            synchronize()
            started[0] = time.perf_counter()

    def model_post(module, inputs, outputs):
        if workflow[0]:
            synchronize()
            times["model"].append(time.perf_counter() - started[0])

    pre_handle, post_handle = (
        model.register_forward_pre_hook(model_pre),
        model.register_forward_hook(model_post),
    )
    oracles, independent = {}, {}
    oracle_calc = (
        MACECalculator(
            model_paths=str(args.checkpoint), device="cpu", default_dtype="float64"
        )
        if args.check_steps and not args.ase_shared_forces
        else None
    )
    errors = {
        "ase_positions": 0.0,
        "ase_cell": 0.0,
        "ase_hessian": 0.0,
        "independent_positions": 0.0,
        "independent_cell": 0.0,
        "independent_hessian": 0.0,
    }
    checked = {}
    update_counts = {} if resume is None else dict(resume["updates"])
    original_pre = bfgs._masked_pre_update

    def update(batch, mask):
        bfgs._ensure_state_initialized(batch)
        ids = batch.system_id.flatten().tolist()
        converged = bfgs._check_convergence(batch)
        excluded = set() if converged is None else set(converged.tolist())
        indices = [
            i
            for i, on in enumerate(mask.tolist())
            if on and i not in excluded and checked.get(ids[i], 0) < args.check_steps
        ]
        workflow[0] = False
        for i in indices:
            identifier = ids[i]
            if identifier not in oracles:
                single = batch.index_select(torch.tensor([i], device=device))
                # Fused BFGS stage code is 1; standalone exit_status is also 1.
                # Give the independent optimizer its own active status domain.
                single.status.zero_()
                atoms = structures[int(batch.source_index[i]) % len(structures)].copy()
                atoms.positions[:] = single.positions.cpu().numpy()
                atoms.cell[:] = single.cell[0].cpu().numpy()
                atoms.calc = oracle_calc
                options = (
                    {
                        "orig_cell": bfgs.filter_states[identifier]
                        .orig_cell.cpu()
                        .numpy()
                    }
                    if args.variable
                    else {}
                )
                target = UnitCellFilter(atoms, **options) if args.variable else atoms
                oracles[identifier] = BFGS(target, logfile=None)
                independent[identifier] = (
                    single,
                    cls(
                        model,
                        backend=args.backend,
                        fmax=args.fmax,
                        hooks=[neighbor()],
                        **options,
                    ),
                )
                state = bfgs.bfgs_states[identifier]
                if state.initialized:
                    # Continue both oracles from the exact paired BFGS history.
                    from ase._4.optimize.bfgs import BFGSMethod

                    oracles[identifier].state = BFGSMethod(
                        state.hessian.cpu().numpy().copy()
                    )
                    oracles[identifier].pos0 = (
                        state.previous_positions.cpu().numpy().ravel().copy()
                    )
                    oracles[identifier].forces0 = (
                        state.previous_forces.cpu().numpy().ravel().copy()
                    )
                    independent[identifier][1].load_state_dict(
                        bfgs.state_dict()["entries"][identifier]["snapshot"],
                        device=device,
                    )
            if args.ase_shared_forces:
                from ase.calculators.singlepoint import SinglePointCalculator
                from ase.stress import full_3x3_to_voigt_6_stress

                oracle = oracles[identifier]
                atoms = oracle.atoms.atoms if args.variable else oracle.atoms
                start, stop = batch.batch_ptr.tolist()[i:i+2]
                values = dict(energy=float(batch.energy[i]),
                              forces=batch.forces[start:stop].cpu().numpy())
                if args.variable:
                    values["stress"] = full_3x3_to_voigt_6_stress(batch.stress[i].cpu().numpy())
                atoms.calc = SinglePointCalculator(atoms, **values)
            oracles[identifier].step()
            single, d = independent[identifier]
            d.step(single)
        workflow[0] = True
        synchronize()
        start = time.perf_counter()
        original_pre(batch, mask)
        synchronize()
        times["bfgs_update"].append(time.perf_counter() - start)
        for i in indices:
            identifier = ids[i]
            a, b = batch.batch_ptr.tolist()[i : i + 2]
            oracle = oracles[identifier]
            atoms = oracle.atoms.atoms if args.variable else oracle.atoms
            single, d = independent[identifier]
            values = dict(
                ase_positions=np.abs(
                    batch.positions[a:b].cpu().numpy() - atoms.positions
                ).max(),
                ase_hessian=np.abs(
                    bfgs.bfgs_states[identifier].hessian.cpu().numpy() - oracle.H
                ).max(),
                independent_positions=(batch.positions[a:b] - single.positions)
                .abs()
                .max()
                .item(),
                independent_hessian=(
                    bfgs.bfgs_states[identifier].hessian - d.bfgs_state.hessian
                )
                .abs()
                .max()
                .item(),
            )
            if args.variable:
                values.update(
                    ase_cell=np.abs(
                        batch.cell[i].cpu().numpy() - atoms.cell.array
                    ).max(),
                    independent_cell=(batch.cell[i : i + 1] - single.cell)
                    .abs()
                    .max()
                    .item(),
                )
            for k, v in values.items():
                errors[k] = max(errors[k], float(v))
            checked[identifier] = checked.get(identifier, 0) + 1
        update_counts.update({i: s.step_count for i, s in bfgs.bfgs_states.items()})

    bfgs._masked_pre_update = update
    run_start = time.monotonic()

    class Progress:
        stage = DynamicsStage.AFTER_STEP
        frequency = 10

        def __call__(self, ctx, event):
            elapsed = time.monotonic() - run_start
            print(
                json.dumps(
                    dict(
                        step=ctx.step_count,
                        elapsed_s=elapsed,
                        status=ctx.batch.status.flatten().tolist(),
                        updates=update_counts,
                    )
                ),
                flush=True,
            )
            if elapsed > args.max_wall_seconds:
                raise TimeoutError("CSP probe wall bound reached")
            if not args.inflight and bool((ctx.batch.status >= 1).all()):
                torch.save(
                    dict(
                        batch=ctx.batch.clone(),
                        bfgs=bfgs.state_dict(),
                        stage_step_count=ctx.step_count + 1,
                        updates=dict(update_counts),
                    ),
                    args.output.with_suffix(".restart.pt"),
                )

    stage.register_hook(Progress())
    profile_state = {"profiler": None, "steps": 0, "finished": False}

    class ProfileStart:
        stage = DynamicsStage.BEFORE_STEP
        frequency = 1

        def __call__(self, ctx, event):
            if (
                not args.profile_bfgs_steps
                or profile_state["finished"]
                or profile_state["profiler"] is not None
            ):
                return
            ids = ctx.batch.system_id.flatten().tolist()
            active = ctx.batch.status.flatten().tolist()
            counts = [
                bfgs.bfgs_states[i].step_count
                for i, status in zip(ids, active)
                if status == 1 and i in bfgs.bfgs_states
            ]
            if counts and min(counts) >= 3:
                synchronize()
                profile = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    + (
                        [torch.profiler.ProfilerActivity.CUDA]
                        if device.type == "cuda"
                        else []
                    ),
                    record_shapes=True,
                )
                profile.__enter__()
                profile_state["profiler"] = profile

    class ProfileEnd:
        stage = DynamicsStage.AFTER_STEP
        frequency = 1

        def __call__(self, ctx, event):
            profile = profile_state["profiler"]
            if profile is None:
                return
            profile_state["steps"] += 1
            if profile_state["steps"] == args.profile_bfgs_steps:
                synchronize()
                profile.__exit__(None, None, None)
                profile.export_chrome_trace(
                    str(args.output.with_suffix(".profile.json"))
                )
                phases = {
                    item.key: dict(
                        calls=item.count,
                        cpu_total_us=item.cpu_time_total,
                        device_total_us=item.device_time_total,
                    )
                    for item in profile.key_averages()
                    if item.key.startswith("bfgs.")
                    or item.key
                    in (
                        "aten::_local_scalar_dense",
                        "aten::clone",
                        "aten::bmm",
                        "nvalchemiops::_bfgs_eigh_hip",
                    )
                }
                args.output.with_suffix(".profile-summary.json").write_text(
                    json.dumps(phases, indent=2) + "\n"
                )
                profile_state["profiler"], profile_state["finished"] = None, True

    stage.register_hook(ProfileStart())
    stage.register_hook(ProfileEnd())
    result = stage.run(
        None
        if args.inflight
        else (resume["batch"] if resume is not None else Batch.from_data_list(rows)),
        n_steps=args.max_steps,
    )
    synchronize()
    elapsed = time.monotonic() - run_start
    pre_handle.remove()
    post_handle.remove()
    final = sink.read() if args.inflight else result
    assert final.num_graphs == len(rows)
    all_converged = bool((final.status == 2).all())
    if args.qualification_only:
        assert all(checked.get(i, 0) >= args.check_steps for i in range(len(rows))), checked
    else:
        assert all_converged, "actual CSP convergence not reached"
    assert (result is None) == args.inflight
    ptr = final.batch_ptr.tolist()
    gfmax = []
    for i, (a, b) in enumerate(zip(ptr, ptr[1:])):
        if args.variable:
            # Default cell_factor=N and full mask: atomic generalized force
            # changes with reference F; check final with retained oracle filter.
            from nvalchemiops.ase_unitcell_filter import ASEUnitCellFilter

            source = int(final.source_index[i]) % len(structures)
            adapter = ASEUnitCellFilter.prepare(
                final.cell[i],
                b - a,
                orig_cell=torch.as_tensor(
                    structures[source].cell.array,
                    dtype=torch.float64,
                    device=final.positions.device,
                ),
            )
            force, _ = adapter.generalized_forces(
                final.forces[a:b], final.stress[i], final.cell[i]
            )
        else:
            force = final.forces[a:b]
        gfmax.append(torch.linalg.vector_norm(force, dim=-1).max().item())
    if not args.qualification_only:
        assert max(gfmax) < args.fmax
    for k, value in errors.items():
        assert value < (2e-6 if "hessian" in k else 2e-8), (k, value)
    summary = dict(
        status="passed_step_qualification" if args.qualification_only else "passed",
        qualification_only=args.qualification_only,
        ase_force_source=("not evaluated" if not args.check_steps else
                          "shared workflow model" if args.ase_shared_forces else
                          "independent CPU MACECalculator"),
        all_converged=all_converged,
        device=str(device),
        torch_version=torch.__version__,
        backend=args.backend,
        variable_cell=args.variable,
        inflight=args.inflight,
        unique_inputs=len(structures),
        input_atoms=[len(a) for a in structures],
        input_elements=[sorted(set(a.get_chemical_symbols())) for a in structures],
        position_perturbation_std_angstrom=args.perturb_positions,
        perturbation_seed_base=7300 if args.perturb_positions else None,
        relaxations=len(rows),
        fmax=args.fmax,
        final_generalized_fmax=gfmax,
        elapsed_s=elapsed,
        stage_steps=stage.step_count,
        updates=update_counts,
        checked_updates=checked,
        errors_max_abs=errors,
        model_evaluations=len(times["model"]),
        comparison_model_evaluations_excluded=True,
        profiled_steps=profile_state["steps"],
        resumed_from=str(args.resume) if args.resume is not None else None,
        timing_s={
            k: dict(total=sum(v), median=float(np.median(v)), samples=len(v))
            for k, v in times.items()
        },
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.cifs
        },
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        system_ids=final.system_id.flatten().tolist(),
        source_indices=final.source_index.flatten().tolist(),
        physical_fmax=[
            torch.linalg.vector_norm(final.forces[a:b], dim=-1).max().item()
            for a, b in zip(ptr, ptr[1:])
        ],
        stress_max_abs=final.stress.abs().amax((-1, -2)).tolist()
        if args.variable
        else None,
        energies=final.energy.flatten().tolist(),
        timing_scope="synchronized model and optimizer timing; total includes independent/ASE comparisons and cold initialization",
        neighbor_backend="torch_reference",
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(final, args.output.with_suffix(".pt"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
