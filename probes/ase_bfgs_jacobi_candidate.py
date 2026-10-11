# SPDX-License-Identifier: Apache-2.0
"""Isolated dynamic HIP Jacobi prototype; no product registration or silent fallback."""

from pathlib import Path

import torch

from nvalchemiops._bfgs_eigh_hip import _check_runtime
from nvalchemiops._hip_extension_loader import load_hip_extension
from nvalchemiops.bfgs_ase import BFGSNumericalError

_EXTENSION = None


def load():
    global _EXTENSION
    _check_runtime()
    if "gfx936" not in torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName:
        raise RuntimeError("prototype qualification requires gfx936")
    if _EXTENSION is None:
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("native"),
            extension_name="ase_bfgs_jacobi_candidate",
            source_names=("ase_jacobi_candidate.cpp", "ase_jacobi_candidate.cu"),
            build_env_var="ASE_JACOBI_PROBE_BUILD_DIR",
            verbose_env_var="ASE_JACOBI_PROBE_BUILD_VERBOSE",
        )
    return _EXTENSION


def validate_metadata(a, max_sweeps):
    if (a.layout != torch.strided or a.ndim not in (2, 3) or a.shape[-2] != a.shape[-1]
            or a.shape[-1] < 1 or not a.is_contiguous()
            or a.dtype != torch.float64 or a.device.type != "cuda" or a.requires_grad
            or (a.ndim == 3 and a.shape[0] < 1)):
        raise ValueError("prototype requires nonempty contiguous non-grad FP64 HIP [n,n] or [B,n,n]")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or not 1 <= max_sweeps <= 100:
        raise ValueError("prototype max_sweeps must be an integer in [1,100]")


def raw_eigh(a, max_sweeps=100, *, chunk_size=None):
    validate_metadata(a, max_sweeps)
    if chunk_size is not None and (isinstance(chunk_size, bool)
                                  or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("chunk_size must be None or a positive integer")
    module = load()
    if a.ndim == 2 or chunk_size is None or chunk_size >= a.shape[0]:
        return module.eigh(a, max_sweeps)
    count, n = a.shape[:2]
    outputs = (a.new_empty((count,n)), a.new_empty(a.shape),
               a.new_empty((count,),dtype=torch.int32), a.new_empty((count,)),
               a.new_empty((count,),dtype=torch.int32))
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        values = module.eigh(a[start:stop], max_sweeps)
        for target, value in zip(outputs, values):
            target[start:stop].copy_(value)
    return outputs


def checked_raw(a, max_sweeps=100, *, chunk_size=None):
    validate_metadata(a, max_sweeps)
    if not bool(torch.isfinite(torch.tril(a)).all()):
        raise BFGSNumericalError("prototype LOWER input contains non-finite entries")
    w, v, info, residual, sweeps = raw_eigh(a, max_sweeps, chunk_size=chunk_size)
    if bool((info != 0).any()):
        raise BFGSNumericalError(f"prototype failed/non-converged: info={info.tolist()}")
    if not bool(torch.isfinite(w).all() & torch.isfinite(v).all() & torch.isfinite(residual).all()):
        raise BFGSNumericalError("prototype returned non-finite eigensystem")
    return w, v, info, residual, sweeps


def eigh(a, UPLO="L", *, chunk_size=None):
    if UPLO != "L":
        raise ValueError("prototype supports only LOWER")
    w, v, _, _, _ = checked_raw(a, chunk_size=chunk_size)
    return w, v
