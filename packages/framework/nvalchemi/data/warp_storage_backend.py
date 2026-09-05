"""Optional NVIDIA Warp implementation of the level-storage backend.

This module is intentionally separate from :mod:`level_storage`: importing the
Torch data model must not import Warp or call ``wp.init()``.  It is loaded only
when a caller explicitly selects the Warp backend in an NVIDIA environment.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from nvalchemi.data import buffer_kernels
from nvalchemi.data.storage_backend import StorageBackend


@wp.kernel(enable_backward=False)
def _expand_segments_kernel(
    starts: wp.array(dtype=Any),
    lengths: wp.array(dtype=Any),
    offsets: wp.array(dtype=Any),
    output: wp.array(dtype=Any),
) -> None:
    """Write each selected segment's element range into ``output``."""
    tid = wp.tid()
    start = starts[tid]
    length = lengths[tid]
    offset = offsets[tid]
    for i in range(wp.int32(length)):
        element = type(start)(i)
        output[offset + element] = start + element


_EXPAND_OVERLOADS: dict[type, Any] = {
    wp.int32: wp.overload(
        _expand_segments_kernel,
        [
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    ),
    wp.int64: wp.overload(
        _expand_segments_kernel,
        [
            wp.array(dtype=wp.int64),
            wp.array(dtype=wp.int64),
            wp.array(dtype=wp.int64),
            wp.array(dtype=wp.int64),
        ],
    ),
}


class WarpStorageBackend(StorageBackend):
    """Adapter preserving the upstream Warp buffer-kernel behavior."""

    name = "warp"

    compute_put_fit_mask_per_system = staticmethod(
        buffer_kernels.compute_put_fit_mask_per_system
    )
    put_masked_per_system = staticmethod(buffer_kernels.put_masked_per_system)
    defrag_per_system = staticmethod(buffer_kernels.defrag_per_system)
    compute_put_fit_mask_segmented = staticmethod(
        buffer_kernels.compute_put_fit_mask_segmented
    )
    put_masked_segmented = staticmethod(buffer_kernels.put_masked_segmented)
    defrag_segmented = staticmethod(buffer_kernels.defrag_segmented)

    def expand_segments(
        self,
        segment_indices: torch.Tensor,
        batch_ptr: torch.Tensor,
        *,
        index_dtype: torch.dtype,
    ) -> torch.Tensor:
        if index_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Unsupported index_dtype {index_dtype}; expected torch.int32 or torch.int64"
            )
        starts = batch_ptr[segment_indices].to(index_dtype)
        ends = batch_ptr[segment_indices + 1].to(index_dtype)
        lengths = ends - starts
        cumulative = torch.cumsum(lengths, dim=0, dtype=index_dtype)
        total = int(cumulative[-1].item()) if cumulative.numel() else 0
        if total > 2**31 - 1:
            raise ValueError(
                f"Total element count {total} exceeds int32 maximum (2147483647); "
                "the Warp kernel uses int32 loop bounds"
            )
        if total == 0:
            return torch.empty(0, device=batch_ptr.device, dtype=index_dtype)
        offsets = cumulative - lengths
        output = torch.empty(total, device=batch_ptr.device, dtype=index_dtype)
        wp_type = wp.int32 if index_dtype is torch.int32 else wp.int64
        wp.launch(
            _EXPAND_OVERLOADS[wp_type],
            dim=segment_indices.numel(),
            inputs=[
                wp.from_torch(starts, dtype=wp_type),
                wp.from_torch(lengths, dtype=wp_type),
                wp.from_torch(offsets, dtype=wp_type),
                wp.from_torch(output, dtype=wp_type),
            ],
            device=f"cuda:{batch_ptr.device.index or 0}"
            if batch_ptr.device.type == "cuda"
            else "cpu",
        )
        return output


__all__ = ["WarpStorageBackend"]
