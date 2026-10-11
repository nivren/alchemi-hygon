# SPDX-License-Identifier: Apache-2.0
"""M5-3B process-local schedulers. No product registration or silent fallback."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ase_bfgs_jacobi_candidate import validate_metadata
from nvalchemiops._bfgs_eigh_hip import _check_runtime, _eigh as rocsolver
from nvalchemiops._hip_extension_loader import load_hip_extension
from nvalchemiops.bfgs_ase import BFGSNumericalError

_EXTENSION = None


def load():
    global _EXTENSION
    _check_runtime()
    if "gfx936" not in torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName:
        raise RuntimeError("M5-3B qualification requires gfx936")
    if _EXTENSION is None:
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("native"),
            extension_name="ase_bfgs_heterogeneous_probe",
            source_names=("ase_jacobi_workers.cpp", "ase_jacobi_candidate.cu", "ase_jacobi_heterogeneous.cu"),
            build_env_var="ASE_HETERO_PROBE_BUILD_DIR", verbose_env_var="ASE_JACOBI_PROBE_BUILD_VERBOSE")
    return _EXTENSION


def check_outputs(outputs):
    for w, v, info, residual, _ in outputs:
        if bool((info != 0).any()):
            raise BFGSNumericalError("heterogeneous probe solver failed/non-converged")
        if not bool(torch.isfinite(w).all() & torch.isfinite(v).all() & torch.isfinite(residual).all()):
            raise BFGSNumericalError("heterogeneous probe returned non-finite eigensystem")
    return outputs


class Scheduler:
    """Persistent, bounded workers; caller stream owns producer/consumer dependency."""
    def __init__(self, variant):
        if variant not in ("rocsolver", "serial", "streams2", "streams4", "native"):
            raise ValueError("unknown experimental schedule")
        self.variant = variant
        self.module = load()
        self.device = torch.cuda.current_device()
        self.workers = int(variant[-1]) if variant.startswith("streams") else 0
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.workers)]
        self.pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers else None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def raw(self, inputs, max_sweeps=100, *, checked=True):
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValueError("nonempty Tensor list required")
        for a in inputs:
            if not isinstance(a, torch.Tensor):
                raise ValueError("Tensor list required")
            validate_metadata(a, max_sweeps)
            if a.device != inputs[0].device or a.device.index != self.device:
                raise ValueError("all groups must use scheduler device")
            if checked and not bool(torch.isfinite(torch.tril(a)).all()):
                raise BFGSNumericalError("heterogeneous LOWER input is non-finite")
        if self.variant == "native":
            result = self.module.heterogeneous(inputs, max_sweeps)
        elif not self.workers:
            if self.variant == "rocsolver":
                # Existing product function already performs finite/info checks.
                if max_sweeps != 100:
                    raise ValueError("rocSOLVER control uses its existing 100-sweep contract")
                result = [rocsolver(a) for a in inputs]
                return result
            result = [self.module.eigh(a, max_sweeps) for a in inputs]
        else:
            if self.pool is None:
                raise RuntimeError("scheduler is closed")
            caller = torch.cuda.current_stream(self.device)
            producer = torch.cuda.Event()
            producer.record(caller)
            bins = [[] for _ in self.streams]
            cost = [0] * self.workers
            for i in sorted(range(len(inputs)), key=lambda i: inputs[i].numel(), reverse=True):
                worker = min(range(self.workers), key=lambda k: cost[k])
                bins[worker].append(i)
                cost[worker] += inputs[i].numel()
            def execute(worker):
                stream = self.streams[worker]
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    stream.wait_event(producer)
                    values = []
                    try:
                        for i in bins[worker]:
                            inputs[i].record_stream(stream)
                            value = self.module.eigh(inputs[i], max_sweeps)
                            if checked:
                                check_outputs([value])
                            values.append((i, value))
                    finally:
                        done = torch.cuda.Event()
                        done.record(stream)
                    return values, done
            futures = [self.pool.submit(execute, k) for k in range(self.workers) if bins[k]]
            result = [None] * len(inputs)
            errors = []
            for future in futures:
                try:
                    values, done = future.result()
                    caller.wait_event(done)
                    for i, value in values:
                        for tensor in value:
                            tensor.record_stream(caller)
                        result[i] = value
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                # A failed worker still owns temporary/input work; drain every
                # stream before propagating, with no state commit by the adapter.
                for stream in self.streams:
                    stream.synchronize()
                raise errors[0]
        return check_outputs(result) if checked and not self.workers else result

    def __call__(self, inputs, max_sweeps=100):
        return [value[:2] for value in self.raw(inputs, max_sweeps)]
