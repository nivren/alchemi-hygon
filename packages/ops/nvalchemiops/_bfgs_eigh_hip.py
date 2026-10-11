# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Private BFGS-local, forward-only FP64 rocSOLVER eigensystem adapter."""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops.backend import BackendUnavailableError
from nvalchemiops.bfgs_ase import BFGSNumericalError

__all__: list[str] = []

_EXTENSION: ModuleType | None = None
_LOCK = threading.Lock()


def _check_runtime() -> None:
    """Admit only the build whose library-internal device path was traced."""
    if (
        getattr(torch.version, "__hcu_version__", None) != "2.9.0+das.opt1.dtk2604"
        or torch.version.hip != "6.3.26093"
    ):
        raise BackendUnavailableError(
            "HIP BFGS eigh requires the verified HCU Torch 2.9.0+das.opt1.dtk2604 "
            "and HIP 6.3.26093 build; other builds require eigensolver requalification"
        )


def _load_extension() -> ModuleType:
    """Lazy-load independent native code; a failed build has no fallback."""
    global _EXTENSION
    _check_runtime()
    with _LOCK:
        if _EXTENSION is None:
            from nvalchemiops._hip_extension_loader import load_hip_extension

            extension = load_hip_extension(
                source_root=Path(__file__).with_name("_native"),
                extension_name="nvalchemi_bfgs_eigh_hip",
                source_names=("bfgs_eigh.cpp", "bfgs_eigh.cu"),
                build_env_var="NVALCHEMI_HIP_BFGS_EIGH_BUILD_DIR",
                verbose_env_var="NVALCHEMI_HIP_BFGS_EIGH_VERBOSE",
                extra_ldflags=("-lrocsolver", "-lrocblas"),
            )
            version = extension.rocsolver_version()
            if version != "3.20.0.80bb870c":
                raise BackendUnavailableError(
                    f"HIP BFGS eigh requires verified rocSOLVER 3.20.0.80bb870c, got {version}"
                )
            _EXTENSION = extension
    return _EXTENSION


def _validate(a: torch.Tensor, UPLO: str, max_sweeps: int) -> None:
    """Validate metadata without device copies or native loading."""
    if a.layout != torch.strided or not a.is_contiguous():
        raise ValueError("BFGS eigh input must be contiguous strided")
    if a.ndim not in (2, 3) or a.shape[-2] != a.shape[-1] or a.numel() == 0:
        raise ValueError("BFGS eigh requires nonempty [n,n] or [B,n,n]")
    if a.dtype != torch.float64:
        raise ValueError("BFGS eigh requires float64")
    if a.requires_grad:
        raise ValueError("BFGS eigh is forward-only; requires_grad is unsupported")
    if a.device.type != "cuda":
        raise BackendUnavailableError("BFGS HIP eigh requires a HIP device")
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO must be L or U")
    if (
        isinstance(max_sweeps, bool)
        or not isinstance(max_sweeps, int)
        or not 0 < max_sweeps <= 2**31 - 1
        or a.shape[-1] > 2**31 - 1
        or (a.ndim == 3 and a.shape[0] > 2**31 - 1)
    ):
        raise ValueError("dimensions and max_sweeps must fit positive int32")


@torch.library.custom_op("nvalchemiops::_bfgs_eigh_hip", mutates_args=())
def _rocsolver_eigh(
    a: torch.Tensor, UPLO: str, max_sweeps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve and check GPU info before returning independent device outputs."""
    _validate(a, UPLO, max_sweeps)
    selected = torch.tril(a) if UPLO == "L" else torch.triu(a)
    if not bool(torch.isfinite(selected).all()):
        raise BFGSNumericalError(
            "BFGS eigh selected triangle contains non-finite input"
        )
    w, v, info, residual, sweeps = _load_extension().eigh(a, UPLO, max_sweeps)
    if bool((info != 0).any()):
        # Diagnostic integers only, never an eigenvector/Hessian CPU transfer.
        raise BFGSNumericalError(
            f"rocSOLVER BFGS eigh failed/non-converged, solver info={info.tolist()}; "
            "state is unchanged"
        )
    if not bool(
        torch.isfinite(w).all()
        & torch.isfinite(v).all()
        & torch.isfinite(residual).all()
    ):
        raise BFGSNumericalError(
            "rocSOLVER BFGS eigh returned non-finite outputs; state is unchanged"
        )
    return w, v, info, residual, sweeps


@_rocsolver_eigh.register_fake
def _fake_eigh(
    a: torch.Tensor, UPLO: str, max_sweeps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Declare contiguous, independent outputs without native/runtime access."""
    _validate(a, UPLO, max_sweeps)
    count = 1 if a.ndim == 2 else a.shape[0]
    return (
        a.new_empty(a.shape[:-1]),
        a.new_empty(a.shape),
        a.new_empty((count,), dtype=torch.int32),
        a.new_empty((count,)),
        a.new_empty((count,), dtype=torch.int32),
    )


def _eigh(a: torch.Tensor, UPLO: str = "L") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the full checked eigensystem using fixed ASE-compatible settings."""
    _validate(a, UPLO, 100)
    w, v, _, _, _ = _rocsolver_eigh(a, UPLO, 100)
    return w, v
