# SPDX-License-Identifier: Apache-2.0
"""Bounded real FIRE2 convergence with explicit neighbor backend and refill.

The hooks only observe/terminate the existing public workflow; no optimizer,
cell filter or neighbor implementation is supplied by this probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read, write

from hip_coo_mace_snapshot import evaluate
from hip_variable_cell_mace_workflow import Dataset, signature
from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, SAMPLES, compare
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FusedStage
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

GPA_PER_EV_A3 = 160.21766208
FMAX = 0.01
STRESS = 0.1 / GPA_PER_EV_A3


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def metrics(values):
    return {
        'energy_eV': float(values['energy'].reshape(-1)[0]),
        'fmax_eV_A': float(values['forces'].norm(dim=-1).max()),
        'max_abs_stress_GPa': float(values['stress'].abs().max()) * GPA_PER_EV_A3,
    }


def physically_converged(values):
    measured = metrics(values)
    return measured['fmax_eV_A'] <= FMAX and measured['max_abs_stress_GPa'] <= 0.1


def joint_convergence():
    return ConvergenceHook(criteria=[
        {'key': 'forces', 'threshold': FMAX, 'reduce_op': 'norm', 'reduce_dims': -1},
        {'key': 'stress', 'threshold': STRESS,
         'custom_op': lambda x: x.abs().amax((-1, -2)) <= STRESS},
    ], source_status=0, target_status=1)


def load_model(checkpoint, dtype, device):
    model = MACEWrapper.from_checkpoint(checkpoint, device=torch.device(device), dtype=dtype,
                                        enable_cueq=False, compile_model=False)
    model.set_config('active_outputs', {'energy', 'forces', 'stress'})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def same_geometry(model, snapshot, dtype):
    hip, he, _ = evaluate(model, snapshot, backend='hip')
    reference, re, _ = evaluate(model, snapshot, backend='torch_reference')
    assert torch.equal(he, re), 'final topology differs between backends'
    return {'parity': compare(hip, reference, dtype), 'edges': len(he),
            'metrics': metrics(reference)}, reference


class Observe:
    stage = DynamicsStage.AFTER_COMPUTE
    frequency = 1

    def __init__(self, model, dtype):
        self.model, self.dtype = model, dtype
        self.calls = 0
        self.admissions = {}
        self.layouts = []

    def __call__(self, ctx, stage):
        batch = ctx.batch
        for key in ('positions', 'cell', 'velocities', 'energy', 'forces', 'stress'):
            value = getattr(batch, key)
            assert value.device == torch.device('cuda:0') and value.dtype == self.dtype, key
            assert bool(torch.isfinite(value).all()), key
        assert bool((torch.linalg.det(batch.cell) > 0).all())
        oracle = batch.clone()
        compute_neighbors(oracle, config=self.model.model_config.neighbor_config,
                          backend='torch_reference')
        assert torch.equal(signature(batch), signature(oracle)), 'current-cell topology differs'
        sources = batch.source_index.flatten().tolist()
        ids = batch.system_id.flatten().tolist()
        ptr = batch.batch_ptr.tolist()
        assert len(sources) <= 2 and batch.num_nodes <= 14
        self.layouts.append({'compute_call': self.calls, 'sources': sources, 'system_ids': ids, 'ptr': ptr})
        new_sources = [i for i, source in enumerate(sources) if source not in self.admissions]
        if new_sources:
            expected, _, _ = evaluate(self.model, batch, backend='torch_reference')
            actual = {key: getattr(batch, key).reshape_as(value) for key, value in expected.items()}
            parity = compare(actual, expected, self.dtype)
            for row in new_sources:
                self.admissions[sources[row]] = {'compute_call': self.calls, 'system_id': ids[row],
                                                'priming_parity': parity}
        self.calls += 1
        if self.calls % 25 == 0:
            print(f'compute {self.calls}: sources={sources}, ptr={ptr}', flush=True)


class CountUpdates:
    stage = DynamicsStage.BEFORE_PRE_UPDATE
    frequency = 1

    def __init__(self):
        self.updates = {}

    def __call__(self, ctx, stage):
        for row, source in enumerate(ctx.batch.source_index.flatten().tolist()):
            if int(ctx.batch.status[row].item()) == 0:
                self.updates[source] = self.updates.get(source, 0) + 1


class Limits:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, updates, max_steps, wall_seconds):
        self.updates, self.max_steps, self.wall_seconds = updates, max_steps, wall_seconds
        self.started = time.monotonic()
        self.exits = {}

    def __call__(self, ctx, stage):
        elapsed = time.monotonic() - self.started
        for row, source in enumerate(ctx.batch.source_index.flatten().tolist()):
            status = int(ctx.batch.status[row].item())
            if status == 0 and elapsed >= self.wall_seconds:
                ctx.batch.status[row] = 3
                status = 3
            elif status == 0 and self.updates.updates[source] >= self.max_steps:
                ctx.batch.status[row] = 2
                status = 2
            if status:
                self.exits[source] = {'status_code': status, 'steps': self.updates.updates[source],
                                      'elapsed_seconds': elapsed}


def run(args):
    if args.max_steps < 1 or args.wall_seconds <= 0:
        raise ValueError('Step and wall budgets must be positive')
    if os.environ.get('HIP_VISIBLE_DEVICES') != '4':
        raise RuntimeError('This acceptance probe requires explicitly allocated HCU4')
    assert torch.version.hip and torch.cuda.is_available()
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    assert sha256(args.checkpoint) == CHECKPOINT_SHA256
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources, systems = [], []
    for name in SAMPLES:
        path = args.data_root / name
        atoms = read(path)
        sources.append({'filename': name, 'sha256': sha256(path), 'atoms': len(atoms)})
        data = AtomicData.from_atoms(atoms, dtype=dtype)
        data.velocities = torch.zeros_like(data.positions)
        data.forces = torch.zeros_like(data.positions)
        data.energy = torch.zeros(1, 1, dtype=dtype)
        data.stress = torch.zeros(1, 3, 3, dtype=dtype)
        systems.append(data)
    model = load_model(args.checkpoint, dtype, 'cuda:0')
    model_loaded = time.monotonic()
    joint = joint_convergence()
    updates = CountUpdates()
    dynamics = FIRE2VariableCell(model=model, dt=0.02, backend='torch_reference',
                                 hooks=[AlignCellHook(backend='torch_reference'), updates],
                                 convergence_hook=joint)
    neighbor = NeighborListHook(model.model_config.neighbor_config,
                                stage=DynamicsStage.BEFORE_COMPUTE, skin=0,
                                backend=args.backend,
                                method='cell_list' if args.backend == 'hip' else None)
    sampler = SizeAwareSampler(Dataset(systems), max_atoms=14, max_batch_size=2)
    sink = HostMemory(capacity=3)
    observer = Observe(model, dtype)
    fused = FusedStage(sub_stages=[(0, dynamics)], hooks=[neighbor, observer],
                       sampler=sampler, sinks=[sink], refill_frequency=1, device_type='cuda')
    # Put the budget observer on FusedStage AFTER_STEP: the existing migration
    # hook must first distinguish physical convergence from budget termination.
    limits = Limits(updates, args.max_steps, args.wall_seconds)
    fused.register_hook(limits)
    initial = sampler.build_initial_batch().to('cuda:0')
    initial_sources = initial.source_index.flatten().tolist()
    initial_raw_cells = {i: data.cell.clone() for i, data in enumerate(systems)}
    dynamics._ensure_state_initialized(initial)
    dynamics._call_hooks(DynamicsStage.BEFORE_STEP, initial)
    outcome = fused.run(batch=initial, n_steps=args.max_steps * len(systems) + 1)
    torch.cuda.synchronize()
    relaxation_end = time.monotonic()
    assert outcome is None and len(sink) == 3, 'not all inputs graduated'
    stored = sink.read()
    assert sorted(stored.source_index.flatten().tolist()) == [0, 1, 2]
    assert sorted(stored.system_id.flatten().tolist()) == [0, 1, 2]
    assert set(observer.admissions) == {0, 1, 2}
    assert any(observer.admissions[source]['compute_call'] > 0 for source in set(range(3)) - set(initial_sources)), 'no refill'
    assert dynamics.step_count == fused.step_count
    cpu_model = load_model(args.checkpoint, dtype, 'cpu')
    records = []
    for data in stored.to_data_list():
        source = int(data.source_index.item())
        snapshot = Batch.from_data_list([data], device='cuda:0')
        cached = {'energy': snapshot.energy.reshape(-1), 'forces': snapshot.forces, 'stress': snapshot.stress}
        audit, reference = same_geometry(model, snapshot, dtype)
        audit['cached_parity'] = compare(cached, reference, dtype)
        status_code = int(data.status.item())
        assert status_code == limits.exits[source]['status_code']
        if status_code == 1:
            assert physically_converged(reference)
        path = args.output_dir / f'{source:02d}-final.cif'
        atoms = Atoms(numbers=data.atomic_numbers.cpu().numpy(),
                      positions=data.positions.cpu().numpy(), cell=data.cell[0].cpu().numpy(), pbc=True)
        write(path, atoms, format='cif')
        reloaded = Batch.from_data_list([AtomicData.from_atoms(read(path), dtype=dtype)])
        c, ce, _ = evaluate(cpu_model, reloaded, backend='torch_reference')
        h, he, _ = evaluate(model, reloaded.to('cuda:0'), backend='torch_reference')
        assert torch.equal(ce.to('cuda:0'), he)
        export_parity = compare(h, {key: value.to('cuda:0') for key, value in c.items()}, dtype)
        if status_code == 1:
            assert physically_converged(c) and physically_converged(h), 'exported CIF fails joint threshold'
        torch.save({key: getattr(data, key).cpu() for key in ('positions', 'cell', 'energy', 'forces', 'stress', 'velocities', 'atomic_numbers')},
                   args.output_dir / f'{source:02d}-final.pt')
        records.append({'source_index': source, 'system_id': int(data.system_id.item()),
                        'filename': sources[source]['filename'], **limits.exits[source],
                        'status': {1: 'converged', 2: 'max_steps', 3: 'wall_timeout'}[status_code],
                        **metrics(reference), 'terminal_audit': audit,
                        'cell_max_abs_change_A': float((data.cell - initial_raw_cells[source]).abs().max()),
                        'final_cif': path.name, 'final_cif_sha256': sha256(path),
                        'export_cpu_hcu_parity': export_parity,
                        'export_cpu_metrics': metrics(c), 'export_hcu_metrics': metrics(h)})
    report = {'status': 'passed' if all(r['status'] == 'converged' for r in records) else 'not_all_converged',
              'product_commit': args.product_commit, 'probe_sha256': sha256(__file__),
              'backend': args.backend, 'optimizer_backend': 'torch_reference', 'dtype': args.dtype,
              'checkpoint_sha256': CHECKPOINT_SHA256, 'sources': sources,
              'criteria': {'fmax_eV_A': FMAX, 'max_abs_stress_GPa': 0.1},
              'limits': {'steps_per_system': args.max_steps, 'wall_seconds': args.wall_seconds},
              'max_active_systems': 2, 'max_atoms': 14, 'dt': 0.02, 'skin': 0,
              'compile_model': False, 'cueq': False,
              'fused_steps': fused.step_count, 'compute_calls': observer.calls,
              'initial_sources': initial_sources, 'admissions': observer.admissions,
              'layouts': observer.layouts, 'records': sorted(records, key=lambda r: r['source_index']),
              'timing': {'load_and_inputs_seconds': model_loaded - started,
                         'relaxation_and_observation_seconds': relaxation_end - model_loaded,
                         'total_seconds': time.monotonic() - started},
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'torch_distribution': importlib.metadata.version('torch'),
                              'hip': torch.version.hip, 'mace': importlib.metadata.version('mace-torch'),
                              'e3nn': importlib.metadata.version('e3nn'),
                              'device': torch.cuda.get_device_name(),
                              'architecture': torch.cuda.get_device_properties(0).gcnArchName,
                              'physical_device': 4, 'sys_executable': sys.executable},
              'scope': 'Three-sample joint convergence/refill and same-geometry terminal parity; no performance or general material guarantee'}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'dtype': args.dtype, 'backend': args.backend,
                      'records': [{key: r[key] for key in ('filename', 'status', 'steps', 'fmax_eV_A', 'max_abs_stress_GPa')} for r in report['records']]}), flush=True)
    return 0 if report['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=['hip', 'torch_reference'], required=True)
    parser.add_argument('--dtype', choices=['float64', 'float32'], required=True)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--wall-seconds', type=float, default=180)
    parser.add_argument('--product-commit', required=True)
    args = parser.parse_args()
    previous = torch.get_default_dtype()
    try:
        code = run(args)
    finally:
        torch.set_default_dtype(previous)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
