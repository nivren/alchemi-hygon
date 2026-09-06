"""Compare framework FIRE2 reference semantics with the locked upstream oracle."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nvalchemi._dynamics_reference.fire import fire2_step_coord


UPSTREAM_FIRE2 = (
    Path(__file__).resolve().parents[4] / "packages/ops/test/dynamics/test_fire2.py"
)


def _load_upstream_reference():
    """Extract only the upstream NumPy oracle without importing Warp tests."""
    tree = ast.parse(UPSTREAM_FIRE2.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fire2_reference_step"
    )
    namespace = {"math": math, "np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(UPSTREAM_FIRE2), "exec"), namespace)  # noqa: S102
    return namespace["_fire2_reference_step"]


def _state(dtype: np.dtype, seed: int):
    rng = np.random.default_rng(seed)
    batch_idx = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    positions = rng.normal(size=(7, 3)).astype(dtype)
    velocities = rng.normal(size=(7, 3)).astype(dtype)
    forces = rng.normal(size=(7, 3)).astype(dtype)
    alpha = np.array([0.09, 0.06, 0.04], dtype=dtype)
    dt = np.array([0.04, 0.03, 0.02], dtype=dtype)
    nsteps_inc = np.array([1, 3, 0], dtype=np.int32)
    return positions, velocities, forces, batch_idx, alpha, dt, nsteps_inc


@pytest.mark.parametrize(
    ("numpy_dtype", "torch_dtype", "rtol", "atol"),
    [
        (np.float32, torch.float32, 5.0e-5, 5.0e-7),
        (np.float64, torch.float64, 1.0e-12, 1.0e-12),
    ],
)
def test_fire2_reference_matches_locked_upstream_numpy(
    numpy_dtype: np.dtype,
    torch_dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    upstream_step = _load_upstream_reference()
    upstream = _state(numpy_dtype, seed=20260906)
    torch_state = tuple(
        (
            tensor.to(dtype=torch_dtype)
            if tensor.is_floating_point()
            else tensor
        )
        for value in upstream
        for tensor in (torch.from_numpy(value.copy()),)
    )
    positions, velocities, forces, batch_idx, alpha, dt, nsteps_inc = torch_state
    kwargs = {
        "delaystep": 2,
        "dtgrow": 1.05,
        "dtshrink": 0.75,
        "alphashrink": 0.985,
        "alpha0": 0.09,
        "tmax": 0.08,
        "tmin": 0.005,
        "maxstep": 0.1,
    }

    for _ in range(5):
        upstream_step(*upstream, **kwargs)
        fire2_step_coord(
            positions,
            velocities,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **kwargs,
        )
        for actual, expected in zip(torch_state, upstream, strict=True):
            np.testing.assert_allclose(
                actual.detach().cpu().numpy(), expected, rtol=rtol, atol=atol
            )
