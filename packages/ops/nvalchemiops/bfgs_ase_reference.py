# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
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

"""Device-local dense BFGS executor, independently matching ASE 3.29 mathematics.

Inputs are validated by nvalchemiops.bfgs_ase before executor invocation. All
tensor arithmetic remains on their device/current Torch stream. Scalar guards
synchronize when required; there is no NumPy, ASE, Warp or CPU tensor fallback.
"""

from collections.abc import Callable, Sequence

import torch

from nvalchemiops.bfgs_ase import ASEBFGSState, ASEBFGSStep, BFGSNumericalError

__all__ = ["bfgs_ase_prepare_state", "bfgs_ase_step", "bfgs_ase_batch_step"]


def bfgs_ase_batch_step(
    positions: Sequence[torch.Tensor],
    forces: Sequence[torch.Tensor],
    states: Sequence[ASEBFGSState],
) -> tuple[ASEBFGSStep, ...]:
    """Use grouped batched arithmetic and the permanent Torch solver reference.

    Parameters
    ----------
    positions, forces : Sequence[torch.Tensor]
        Independent contiguous FP64 generalized coordinates and forces.
    states : Sequence[ASEBFGSState]
        Per-system state validated by the public dispatcher.

    Returns
    -------
    tuple[ASEBFGSStep, ...]
        Per-system diagnostics in input order, after all groups pass validation.
    """
    from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh

    return _step_batch_with_eigh(positions, forces, states, torch.linalg.eigh)


def bfgs_ase_prepare_state(
    num_scalar_dofs: int,
    *,
    alpha: float,
    maxstep: float,
    dtype: torch.dtype,
    device: torch.device,
    system_id: int | None,
    coordinate_mode: str = "cartesian",
) -> ASEBFGSState:
    """Allocate dense Hessian and independent history on the selected device."""
    history = torch.zeros((num_scalar_dofs // 3, 3), dtype=dtype, device=device)
    return ASEBFGSState(
        torch.eye(num_scalar_dofs, dtype=dtype, device=device) * alpha,
        history,
        torch.zeros_like(history),
        alpha=alpha,
        maxstep=maxstep,
        system_id=system_id,
        coordinate_mode=coordinate_mode,
    )


def _finite(name: str, tensor: torch.Tensor) -> None:
    """Fail before commit when the unmodified ASE formula is non-finite."""
    if not bool(torch.isfinite(tensor).all()):
        raise BFGSNumericalError(
            f"ASE BFGS produced non-finite {name}; state is unchanged"
        )


def _check_eigensystem(hessian, eigenvalues, vectors):
    """Reject finite but incorrect solver output without altering its spectrum.

    The admitted DTK DC path can return info=0 with a wrong eigensystem.
    Checks run on the input device. Frobenius reconstruction is relative to H;
    orthogonality is relative to the identity norm (sqrt(d)). FP64 limit 1e-10
    is a solver validity guard, not a curvature/spectrum modification.
    """
    lower = torch.tril(hessian)
    symmetric = lower + torch.tril(hessian, diagonal=-1).mT
    residual = torch.linalg.matrix_norm(
        symmetric @ vectors - vectors * eigenvalues.unsqueeze(-2)
    ) / torch.linalg.matrix_norm(symmetric).clamp_min(1)
    width = hessian.shape[-1]
    identity = torch.eye(width, dtype=hessian.dtype, device=hessian.device)
    orthogonal = torch.linalg.matrix_norm(vectors.mT @ vectors - identity) / (
        width**0.5
    )
    if not bool(
        (
            torch.isfinite(residual)
            & torch.isfinite(orthogonal)
            & (residual <= 1e-10)
            & (orthogonal <= 1e-10)
        ).all()
    ):
        raise BFGSNumericalError(
            "ASE BFGS invalid eigensystem residual/orthogonality; state is unchanged"
        )


def bfgs_ase_step(
    positions: torch.Tensor, forces: torch.Tensor, state: ASEBFGSState
) -> ASEBFGSStep:
    """Use the permanent Torch eigensolver reference with shared ASE math."""
    return _step_with_eigh(positions, forces, state, torch.linalg.eigh)


def _step_with_eigh(
    positions: torch.Tensor,
    forces: torch.Tensor,
    state: ASEBFGSState,
    eigh: Callable,
) -> ASEBFGSStep:
    """Compute an exact ASE-style candidate and then commit one finite step.

    No curvature skip, damping, spectrum clamp or extra near-zero threshold
    is applied. History contains the coordinates before this step.
    """
    with torch.profiler.record_function("bfgs.hessian_update"):
        hessian = state.hessian
        if state.initialized:
            displacement = (positions - state.previous_positions).reshape(-1)
            if bool(displacement.abs().max() >= 1e-7):
                gradient_delta = (state.previous_forces - forces).reshape(-1)
                hs = hessian @ displacement
                curvature = displacement @ gradient_delta
                hcurvature = displacement @ hs
                if bool(curvature == 0) or bool(hcurvature == 0):
                    raise BFGSNumericalError(
                        "ASE BFGS exact zero update denominator; state is unchanged"
                    )
                hessian = hessian + (
                    torch.outer(gradient_delta, gradient_delta) / curvature
                    - torch.outer(hs, hs) / hcurvature
                )
        _finite("Hessian", hessian)
    try:
        with torch.profiler.record_function("bfgs.eigh"):
            eigenvalues, vectors = eigh(hessian, UPLO="L")
    except torch.linalg.LinAlgError as exc:
        raise BFGSNumericalError(
            "ASE BFGS eigendecomposition failed; state is unchanged"
        ) from exc
    _finite("eigenvalues", eigenvalues)
    if bool((eigenvalues == 0).any()):
        raise BFGSNumericalError("ASE BFGS exact zero eigenvalue; state is unchanged")
    _check_eigensystem(hessian, eigenvalues, vectors)
    raw = (vectors @ ((forces.reshape(-1) @ vectors) / eigenvalues.abs())).reshape_as(
        positions
    )
    _finite("direction", raw)
    length = torch.linalg.vector_norm(raw, dim=1).max()
    _finite("step length", length)
    scale = torch.ones((), dtype=positions.dtype, device=positions.device)
    if bool(length >= state.maxstep):
        scale = state.maxstep / length
    applied = raw * scale
    next_positions = positions + applied
    _finite("coordinates", next_positions)
    state.previous_positions.copy_(positions)
    state.previous_forces.copy_(forces)
    state.hessian.copy_(hessian)
    positions.copy_(next_positions)
    state.initialized = True
    state.step_count += 1
    return ASEBFGSStep(eigenvalues, raw, applied, scale, length)
