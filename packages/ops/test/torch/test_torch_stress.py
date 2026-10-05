# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent row-cell energy-gradient and interface checks for cell forces."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.stress_reference import stress_to_cell_force


def inputs(dtype=torch.float64):
    """Return two unequal nonorthogonal systems with periodic bond vectors."""
    h = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.6, 2.4, 0.0], [-0.3, 0.5, 2.8]],
            [[2.6, 0.2, -0.4], [0.7, 2.2, 0.3], [-0.2, 0.6, 3.1]],
        ],
        dtype=dtype,
    )
    q = torch.tensor(
        [[[1.1, -0.2, 0.4], [0.2, 0.7, -0.1]], [[0.4, 0.2, 1.2], [-0.3, 0.8, 0.2]]],
        dtype=dtype,
    )
    r = q @ h
    v = torch.linalg.det(h).abs()
    sigma = r.mT @ r / v[:, None, None]
    return sigma, h, v, q


@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_periodic_harmonic_cell_energy_gradient(dtype, aligned):
    """F=-dE/dH must agree with an independent periodic harmonic energy."""
    s, h, v, q = inputs(dtype)
    result = stress_to_cell_force(s, h, v, aligned)
    expected = -(q.mT @ (q @ h))
    if aligned:
        expected = torch.tril(expected)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    if dtype == torch.float64:
        fd = torch.zeros_like(h)
        eps = 1e-6
        for b in range(2):
            for i in range(3):
                for j in range(3):
                    delta = torch.zeros_like(h[b])
                    delta[i, j] = eps
                    ep = 0.5 * (q[b] @ (h[b] + delta)).square().sum()
                    em = 0.5 * (q[b] @ (h[b] - delta)).square().sum()
                    fd[b, i, j] = -(ep - em) / (2 * eps)
        if aligned:
            fd = torch.tril(fd)
        torch.testing.assert_close(result, fd, rtol=2e-8, atol=2e-8)
        wrong = -v[:, None, None] * (s @ torch.linalg.inv(h).mT)
        if aligned:
            wrong = torch.tril(wrong)
        assert (wrong - expected).abs().max() > 0.1


def test_strided_batch_matches_independent_outputs_and_does_not_alias():
    """Per-system conversion accepts strides and owns its output storage."""
    s, h, v, _ = inputs()
    s, h = s.mT, h.mT
    before = [x.clone() for x in (s, h, v)]
    result = stress_to_cell_force(s, h, v, False)
    single = torch.cat(
        [
            stress_to_cell_force(s[i : i + 1], h[i : i + 1], v[i : i + 1], False)
            for i in range(2)
        ]
    )
    torch.testing.assert_close(result, single)
    assert result.is_contiguous()
    result.zero_()
    for x, old in zip((s, h, v), before, strict=True):
        assert torch.equal(x, old)


@pytest.mark.parametrize("aligned", [False, True])
def test_first_and_second_derivatives(aligned):
    """Stress, cell and independently supplied volume remain twice differentiable."""
    s, h, v, _ = inputs()
    args = tuple(x.requires_grad_() for x in (s, h, v))

    def fn(*xs):
        return stress_to_cell_force(*xs, aligned)

    assert torch.autograd.gradcheck(fn, args)
    assert torch.autograd.gradgradcheck(fn, args)


def test_empty_and_zero_stress():
    """Empty inputs retain gradient connectivity; zero stress produces zero force."""
    s, h, v, _ = inputs()
    assert torch.count_nonzero(stress_to_cell_force(torch.zeros_like(s), h, v)) == 0
    empty = tuple(x[:0].requires_grad_() for x in (s, h, v))
    result = stress_to_cell_force(*empty)
    assert result.shape == (0, 3, 3)
    grads = torch.autograd.grad(result.sum(), empty)
    assert all(g.shape == x.shape for g, x in zip(grads, empty, strict=True))


@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "nan", "volume", "singular", "float16"]
)
def test_invalid_inputs_fail_explicitly(kind):
    """Reject bad contracts rather than coercing or silently returning invalid force."""
    s, h, v, _ = inputs()
    aligned = True
    if kind == "shape":
        s = s[:, 0]
    elif kind == "dtype":
        v = v.float()
    elif kind == "nan":
        s[0, 0, 0] = float("nan")
    elif kind == "volume":
        v[0] = 0
    elif kind == "singular":
        h[0, 1] = h[0, 0]
    else:
        s, h, v = s.half(), h.half(), v.half()
    with pytest.raises((TypeError, ValueError)):
        stress_to_cell_force(s, h, v, aligned)


def test_custom_op_and_cpu_compile():
    """Verify schema, fake/meta, autograd registration and AOT execution."""
    s, h, v, _ = inputs()
    args = tuple(x.requires_grad_() for x in (s, h, v))
    torch.library.opcheck(torch.ops.nvalchemiops.stress_to_cell_force.default, args)
    compiled = torch.compile(stress_to_cell_force, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(*args, False), stress_to_cell_force(*args, False)
    )
