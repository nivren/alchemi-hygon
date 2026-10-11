# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Grouped device-local ASE BFGS mathematics, with all-group atomic commit."""

import torch

from nvalchemiops.bfgs_ase import ASEBFGSStep, BFGSNumericalError
from nvalchemiops.bfgs_ase_reference import _check_eigensystem, _finite


def _step_batch_with_eigh(positions, forces, states, eigh, *, grouped_eigh=None):
    """Stack equal DOFs; vectorize arithmetic; commit only validated candidates."""
    groups = {}
    for i, state in enumerate(states):
        groups.setdefault(state.num_scalar_dofs, []).append(i)
    candidates = {}
    prepared = []
    for width, indices in groups.items():
        q = torch.stack([positions[i] for i in indices])
        f = torch.stack([forces[i] for i in indices])
        h = torch.stack([states[i].hessian for i in indices])
        oldq = torch.stack([states[i].previous_positions for i in indices])
        oldf = torch.stack([states[i].previous_forces for i in indices])
        with torch.profiler.record_function("bfgs.batch.hessian_update"):
            s = (q - oldq).reshape(-1, width)
            dg = (oldf - f).reshape(-1, width)
            initialized = torch.tensor(
                [states[i].initialized for i in indices],
                device=q.device,
                dtype=torch.bool,
            )
            update = initialized & (s.abs().amax(-1) >= 1e-7)
            hs = torch.bmm(h, s.unsqueeze(-1)).squeeze(-1)
            curvature = (s * dg).sum(-1)
            hc = (s * hs).sum(-1)
            if bool((update & ((curvature == 0) | (hc == 0))).any()):
                raise BFGSNumericalError(
                    "ASE BFGS exact zero update denominator; batch unchanged"
                )
            # Inactive histories do not divide by their uninitialized zero denominators.
            c = torch.where(update, curvature, torch.ones_like(curvature))
            b = torch.where(update, hc, torch.ones_like(hc))
            delta = (
                dg[:, :, None] * dg[:, None, :] / c[:, None, None]
                - hs[:, :, None] * hs[:, None, :] / b[:, None, None]
            )
            h = h + torch.where(update[:, None, None], delta, torch.zeros_like(delta))
            _finite("Hessian", h)
        prepared.append((width, indices, q, f, h))
    try:
        with torch.profiler.record_function("bfgs.batch.eigh"):
            decompositions = (
                grouped_eigh([row[4] for row in prepared])
                if grouped_eigh is not None
                else [eigh(row[4], UPLO="L") for row in prepared]
            )
    except torch.linalg.LinAlgError as exc:
        raise BFGSNumericalError(
            "ASE BFGS eigendecomposition failed; batch unchanged"
        ) from exc
    if len(decompositions) != len(prepared):
        raise BFGSNumericalError("incomplete group decomposition; batch unchanged")
    for (width, indices, q, f, h), (w, v) in zip(prepared, decompositions):
        _finite("eigenvalues", w)
        if bool((w == 0).any()):
            raise BFGSNumericalError("ASE BFGS exact zero eigenvalue; batch unchanged")
        _check_eigensystem(h, w, v)
        projected = torch.bmm(f.reshape(-1, 1, width), v).squeeze(1)
        raw = torch.bmm(v, (projected / w.abs()).unsqueeze(-1)).reshape_as(q)
        _finite("direction", raw)
        length = torch.linalg.vector_norm(raw, dim=-1).amax(-1)
        _finite("step length", length)
        limits = torch.tensor(
            [states[i].maxstep for i in indices], dtype=q.dtype, device=q.device
        )
        denominator = torch.where(length >= limits, length, torch.ones_like(length))
        scale = torch.where(
            length >= limits, limits / denominator, torch.ones_like(length)
        )
        applied = raw * scale[:, None, None]
        newq = q + applied
        _finite("coordinates", newq)
        for j, i in enumerate(indices):
            candidates[i] = (
                newq[j],
                h[j],
                ASEBFGSStep(
                    w[j].clone(),
                    raw[j].clone(),
                    applied[j].clone(),
                    scale[j].clone(),
                    length[j].clone(),
                ),
            )
    for i, state in enumerate(states):
        newq, h, result = candidates[i]
        state.previous_positions.copy_(positions[i])
        state.previous_forces.copy_(forces[i])
        state.hessian.copy_(h)
        positions[i].copy_(newq)
        state.initialized = True
        state.step_count += 1
    return tuple(candidates[i][2] for i in range(len(states)))
