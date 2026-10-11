# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Private FP64 Jacobi custom op and bounded per-device group scheduling."""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops._bfgs_eigh_hip import _check_runtime
from nvalchemiops.backend import BackendUnavailableError
from nvalchemiops.bfgs_ase import BFGSNumericalError

__all__: list[str] = []

_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()
_SCHEDULERS: dict[torch.device, _GroupScheduler] = {}
_SCHEDULERS_LOCK = threading.Lock()


def _load_extension(device: torch.device) -> ModuleType:
    """Load before worker creation; admit only the qualified build/architecture."""
    global _EXTENSION
    _check_runtime()
    with torch.cuda.device(device):
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch != "gfx936":
            raise BackendUnavailableError(
                "HIP BFGS Jacobi requires qualified gfx936; requalification required"
            )
        with _EXTENSION_LOCK:
            if _EXTENSION is None:
                from nvalchemiops._hip_extension_loader import load_hip_extension

                _EXTENSION = load_hip_extension(
                    source_root=Path(__file__).with_name("_native"),
                    extension_name="nvalchemi_bfgs_jacobi_hip",
                    source_names=("bfgs_jacobi.cpp", "bfgs_jacobi.cu"),
                    build_env_var="NVALCHEMI_HIP_BFGS_JACOBI_BUILD_DIR",
                    verbose_env_var="NVALCHEMI_HIP_BFGS_JACOBI_VERBOSE",
                )
    return _EXTENSION


def _validate(a: torch.Tensor, UPLO: str, max_sweeps: int) -> None:
    """Validate native representation limits without loading or device copies."""
    if not isinstance(a, torch.Tensor):
        raise ValueError("BFGS Jacobi requires Tensor input")
    if a.layout != torch.strided or not a.is_contiguous():
        raise ValueError("BFGS Jacobi requires contiguous strided input")
    if a.ndim not in (2, 3) or a.shape[-2] != a.shape[-1] or a.numel() == 0:
        raise ValueError("BFGS Jacobi requires nonempty [n,n] or [B,n,n]")
    if a.dtype != torch.float64:
        raise ValueError("BFGS Jacobi requires float64")
    if a.requires_grad:
        raise ValueError("BFGS Jacobi is forward-only; requires_grad unsupported")
    if a.device.type != "cuda":
        raise BackendUnavailableError("BFGS Jacobi requires a HIP device")
    if UPLO != "L":
        raise ValueError("BFGS Jacobi supports only LOWER (UPLO='L')")
    if (
        isinstance(max_sweeps, bool)
        or not isinstance(max_sweeps, int)
        or not 1 <= max_sweeps <= 100
        or a.shape[-1] >= 2**31 - 1
        or a.numel() > 2**63 - 1
    ):
        raise ValueError(
            "BFGS Jacobi unrepresentable size or sweep limit outside 1..100"
        )


@torch.library.custom_op("nvalchemiops::_bfgs_jacobi_hip", mutates_args=())
def _jacobi_eigh(
    a: torch.Tensor, max_sweeps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return checked, independent device outputs; control transfers are scalar."""
    _validate(a, "L", max_sweeps)
    if not bool(torch.isfinite(torch.tril(a)).all()):
        raise BFGSNumericalError("BFGS Jacobi LOWER input contains non-finite entries")
    w, v, info, residual, sweeps = _load_extension(a.device).eigh(a, max_sweeps)
    if bool((info != 0).any()):
        raise BFGSNumericalError(
            f"BFGS Jacobi failed/non-converged, info={info.tolist()}; state unchanged"
        )
    if not bool(
        torch.isfinite(w).all()
        & torch.isfinite(v).all()
        & torch.isfinite(residual).all()
    ):
        raise BFGSNumericalError(
            "BFGS Jacobi returned non-finite outputs; state unchanged"
        )
    return w, v, info, residual, sweeps


@_jacobi_eigh.register_fake
def _fake_jacobi(
    a: torch.Tensor, max_sweeps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Declare independent output metadata without touching runtime or native code."""
    _validate(a, "L", max_sweeps)
    count = 1 if a.ndim == 2 else a.shape[0]
    return (
        a.new_empty(a.shape[:-1]),
        a.new_empty(a.shape),
        a.new_empty((count,), dtype=torch.int32),
        a.new_empty((count,)),
        a.new_empty((count,), dtype=torch.int32),
    )


def _eigh(a: torch.Tensor, UPLO: str = "L") -> tuple[torch.Tensor, torch.Tensor]:
    """Solve on the caller's current stream with ASE-compatible fixed settings."""
    _validate(a, UPLO, 100)
    return _jacobi_eigh(a, 100)[:2]


class _GroupScheduler:
    """Two persistent workers on one device; serialized calls and explicit close."""

    def __init__(self, device: torch.device):
        self.device = device
        _load_extension(device)
        with torch.cuda.device(device):
            self.streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bfgs-jacobi")
        self.lock = threading.Lock()
        self.closed = False

    def close(self) -> None:
        """Wait for any submitted call before shutting down the owned worker pool."""
        with self.lock:
            if not self.closed:
                self.pool.shutdown(wait=True)
                self.closed = True

    def solve(self, inputs: list[torch.Tensor], max_sweeps: int = 100):
        """Preserve input order, stream dependencies and allocator lifetimes."""
        if not inputs:
            raise ValueError("BFGS Jacobi requires nonempty group list")
        for a in inputs:
            _validate(a, "L", max_sweeps)
            if a.device != self.device:
                raise ValueError("all BFGS Jacobi groups must use the scheduler device")
        with self.lock:
            if self.closed:
                raise RuntimeError("BFGS Jacobi scheduler is closed")
            caller = torch.cuda.current_stream(self.device)
            producer = torch.cuda.Event()
            producer.record(caller)
            bins, costs = [[], []], [0, 0]
            for i in sorted(
                range(len(inputs)), key=lambda j: inputs[j].numel(), reverse=True
            ):
                worker = min(range(2), key=lambda j: costs[j])
                bins[worker].append(i)
                costs[worker] += inputs[i].numel()

            def execute(worker):
                stream = self.streams[worker]
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    stream.wait_event(producer)
                    values = []
                    for i in bins[worker]:
                        inputs[i].record_stream(stream)
                        values.append((i, _jacobi_eigh(inputs[i], max_sweeps)))
                    done = torch.cuda.Event()
                    done.record(stream)
                    return values, done

            futures = []
            errors = []
            try:
                futures.extend(
                    self.pool.submit(execute, k) for k in range(2) if bins[k]
                )
            except BaseException as exc:
                # A submission failure must also drain work already accepted.
                errors.append(exc)
            result = [None] * len(inputs)
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
                for future in futures:
                    if not future.done():
                        try:
                            future.result()
                        except BaseException as exc:
                            errors.append(exc)
                for stream in self.streams:
                    stream.synchronize()
                raise errors[0]
            return result


def _grouped_eigh(
    inputs: list[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Use direct batched solve for one width; cache two workers for multiple widths."""
    if not inputs:
        raise ValueError("BFGS Jacobi requires nonempty group list")
    for a in inputs:
        _validate(a, "L", 100)
    device = inputs[0].device
    if any(a.device != device for a in inputs):
        raise ValueError("all BFGS Jacobi groups must use one device")
    if len(inputs) == 1:
        return [_eigh(inputs[0])]
    with _SCHEDULERS_LOCK:
        scheduler = _SCHEDULERS.get(device)
        if scheduler is None:
            scheduler = _GroupScheduler(device)
            _SCHEDULERS[device] = scheduler
    return [value[:2] for value in scheduler.solve(inputs)]


def _close_schedulers() -> None:
    """Release cached workers explicitly or at process exit; allow later recreation."""
    with _SCHEDULERS_LOCK:
        for scheduler in _SCHEDULERS.values():
            scheduler.close()
        _SCHEDULERS.clear()


atexit.register(_close_schedulers)
