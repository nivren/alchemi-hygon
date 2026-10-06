# SPDX-License-Identifier: Apache-2.0
"""Batch metadata exact integer oracle before native query integration."""
import pytest
import torch

from nvalchemiops import _hip_neighbor_executor as hip
from nvalchemiops.torch_reference_cell_list import _grid_spec


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires allocated HIP device"))]


def legacy(cells, cutoff, limit):
    rows = [_grid_spec(cell, cutoff, max_nbins=limit, min_cells_per_dimension=1) for cell in cells]
    dims, radius = [torch.stack([row[i] for row in rows]) for i in (0, 1)]
    sizes = dims.long().prod(-1)
    offsets = torch.zeros(len(cells), device=cells.device, dtype=torch.int32)
    offsets[1:] = sizes.cumsum(0)[:-1].int()
    return dims, radius, offsets, torch.linalg.inv(cells)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["ordinary", "rotate", "compression", "shear", "reduction", "huge", "cutoff_boundary", "empty", "singular"])
def test_metadata_against_per_system_oracle(case, dtype, device, monkeypatch):
    cells = torch.tensor([[[12., 0, 0], [.3, 13, 0], [.2, .4, 14]],
                          [[6., .5, .1], [.7, 7, .2], [.3, .2, 8]],
                          [[100., 0, 0], [0, 100, 0], [0, 0, 100]]], dtype=dtype, device=device)
    cutoff = 2.0
    limit = 8192
    if case == "rotate":
        cells = cells @ cells.new_tensor([[.8, -.6, 0], [.6, .8, 0], [0, 0, 1]])
    elif case == "compression":
        cells *= .23
    elif case == "shear":
        cells = cells @ cells.new_tensor([[1, .4, 0], [.1, 1, .2], [.2, -.1, 1]])
    elif case == "reduction":
        limit = 25
    elif case == "huge":
        cells *= 1e7  # Original Python product is unbounded; int64 product would overflow.
    elif case == "cutoff_boundary":
        cells = torch.diag_embed(cells.new_tensor([[6., 12., 24.]]).expand(3, -1)).clone()
        cells[0, 0, 0] = torch.nextafter(cells[0, 0, 0], cells.new_tensor(float("-inf")))
        cells[2, 0, 0] = torch.nextafter(cells[2, 0, 0], cells.new_tensor(float("inf")))
        cutoff = 6.
    elif case == "empty":
        cells = cells[:0]
    elif case == "singular":
        cells[1, 1].zero_()  # Exact zero pivot in both dtypes, independent of LU rounding.
    monkeypatch.setattr(hip, "_MAX_NBINS", limit)
    if case == "singular":
        with pytest.raises(RuntimeError, match="volume == 0.0"):
            hip._current_cell_metadata(cells, cutoff)
        return
    actual = hip._current_cell_metadata(cells, cutoff)
    if case == "empty":
        assert [tuple(t.shape) for t in actual] == [(0, 3), (0, 3), (0,), (0, 3, 3)]
        return
    expected = legacy(cells, cutoff, limit)
    for a, b in zip(actual[:3], expected[:3]):
        assert torch.equal(a, b)
    tol = 2e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(actual[3], expected[3], rtol=tol, atol=tol)
    assert actual[3].is_contiguous()
