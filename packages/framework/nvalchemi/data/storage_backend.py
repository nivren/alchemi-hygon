"""Backend contract for level-storage mutation and segmented indexing.

The data model stays Torch-native.  Implementations of this protocol may use
Torch reference operations, Triton, HIP, or Warp, but the protocol itself must
not import any accelerator-specific runtime.  Keeping this boundary import
light is required so ``AtomicData`` and the public data namespace can be used
without initializing NVIDIA Warp.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch

StorageBackendName = Literal["auto", "torch", "triton", "hip", "warp"]
_INT32_MAX = 2**31 - 1


def _check_same_device(*tensors: torch.Tensor) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"Storage backend tensors must share one device; got {sorted(map(str, devices))}")


def _as_rows(tensor: torch.Tensor) -> torch.Tensor:
    """View a row-oriented tensor as ``(rows, -1)`` without changing storage."""
    return tensor.reshape(tensor.shape[0], -1)


class StorageBackend(Protocol):
    """Operations required by the level-storage containers.

    Methods mutate output tensors in place to preserve the upstream storage
    contract.  Implementations must keep all tensors on their existing device,
    preserve dtype and shape rules, and report unsupported input combinations
    instead of silently moving data or changing precision.
    """

    name: str

    def compute_put_fit_mask_per_system(
        self,
        source_mask: torch.Tensor,
        dest_mask: torch.Tensor,
        fit_mask: torch.Tensor,
    ) -> None:
        """Mark source rows that fit into available destination slots."""

    def put_masked_per_system(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        dest: torch.Tensor,
        dest_mask: torch.Tensor,
        source_data_copied: torch.Tensor,
    ) -> None:
        """Copy fitting masked rows and update both masks in place."""

    def defrag_per_system(
        self,
        source: torch.Tensor,
        source_data_copied: torch.Tensor,
    ) -> None:
        """Compact kept rows in place and update the copied mask."""

    def compute_put_fit_mask_segmented(
        self,
        source_batch_ptr: torch.Tensor,
        source_mask: torch.Tensor,
        dest_batch_ptr: torch.Tensor,
        num_dest_segments: int,
        dest_capacity: int,
        fit_mask: torch.Tensor,
    ) -> None:
        """Mark source segments that fit in data and pointer capacities."""

    def put_masked_segmented(
        self,
        source: torch.Tensor,
        source_batch_ptr: torch.Tensor,
        source_mask: torch.Tensor,
        dest: torch.Tensor,
        dest_batch_ptr: torch.Tensor,
        num_dest_segments: int,
        source_data_copied: torch.Tensor,
    ) -> torch.Tensor | None:
        """Copy fitting segments and append their boundaries in place."""

    def defrag_segmented(
        self,
        source: torch.Tensor,
        source_batch_ptr: torch.Tensor,
        source_data_copied: torch.Tensor,
    ) -> torch.Tensor:
        """Compact kept segments and return the number of kept segments."""

    def expand_segments(
        self,
        segment_indices: torch.Tensor,
        batch_ptr: torch.Tensor,
        *,
        index_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand selected segment indices to element indices."""


class TorchStorageBackend:
    """Torch reference implementation of the level-storage mutation contract.

    This backend intentionally uses only Torch operations and never imports
    Warp.  It is the semantic baseline for HCU execution; Triton and HIP can
    later replace individual methods without changing the storage classes.
    Dynamic counts use a small host read where the public API requires a
    Python-sized slice or returned tensor.  No data is moved to CPU.
    """

    name = "torch"

    def compute_put_fit_mask_per_system(
        self,
        source_mask: torch.Tensor,
        dest_mask: torch.Tensor,
        fit_mask: torch.Tensor,
    ) -> None:
        _check_same_device(source_mask, dest_mask, fit_mask)
        if source_mask.numel() == 0:
            return
        if source_mask.dtype is not torch.bool or dest_mask.dtype is not torch.bool:
            raise TypeError("source_mask and dest_mask must have dtype torch.bool")
        if fit_mask.shape != source_mask.shape or fit_mask.dtype is not torch.bool:
            raise ValueError("fit_mask must be a bool tensor with source_mask shape")
        available = (~dest_mask).sum(dtype=torch.int64)
        rank = torch.cumsum(source_mask.to(torch.int64), dim=0) - source_mask.to(torch.int64)
        fit_mask.copy_(source_mask & (rank < available))

    def put_masked_per_system(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        dest: torch.Tensor,
        dest_mask: torch.Tensor,
        source_data_copied: torch.Tensor,
    ) -> None:
        _check_same_device(source, source_mask, dest, dest_mask, source_data_copied)
        if source.shape[0] == 0:
            return
        if source_mask.dtype is not torch.bool or dest_mask.dtype is not torch.bool:
            raise TypeError("source_mask and dest_mask must have dtype torch.bool")
        if source_data_copied.shape != source_mask.shape or source_data_copied.dtype is not torch.bool:
            raise ValueError("source_data_copied must be a bool tensor with source_mask shape")
        source_rows, dest_rows = _as_rows(source), _as_rows(dest)
        if source_rows.shape[1] != dest_rows.shape[1]:
            raise ValueError("source and dest must have matching trailing element counts")

        source_data_copied.zero_()
        source_indices = torch.nonzero(source_mask, as_tuple=False).flatten()
        dest_indices = torch.nonzero(~dest_mask, as_tuple=False).flatten()
        count = min(source_indices.numel(), dest_indices.numel())
        if count == 0:
            return
        source_indices = source_indices[:count]
        dest_indices = dest_indices[:count]
        values = source_rows.index_select(0, source_indices).clone()
        dest_rows.index_copy_(0, dest_indices, values)
        dest_mask.index_fill_(0, dest_indices, True)
        source_data_copied.index_fill_(0, source_indices, True)

    def defrag_per_system(
        self,
        source: torch.Tensor,
        source_data_copied: torch.Tensor,
    ) -> None:
        _check_same_device(source, source_data_copied)
        if source.shape[0] == 0:
            return
        if source_data_copied.dtype is not torch.bool or source_data_copied.ndim != 1:
            raise ValueError("source_data_copied must be a one-dimensional bool tensor")
        if source_data_copied.shape[0] != source.shape[0]:
            raise ValueError("source_data_copied length must equal source rows")
        source_rows = _as_rows(source)
        keep = ~source_data_copied
        kept = source_rows[keep].clone()
        num_kept = kept.shape[0]
        source.zero_()
        if num_kept:
            _as_rows(source)[:num_kept].copy_(kept)
        source_data_copied.zero_()
        source_data_copied[num_kept:].fill_(True)

    def compute_put_fit_mask_segmented(
        self,
        source_batch_ptr: torch.Tensor,
        source_mask: torch.Tensor,
        dest_batch_ptr: torch.Tensor,
        num_dest_segments: int,
        dest_capacity: int,
        fit_mask: torch.Tensor,
    ) -> None:
        _check_same_device(source_batch_ptr, source_mask, dest_batch_ptr, fit_mask)
        num_systems = source_batch_ptr.shape[0] - 1
        if num_systems == 0:
            return
        if source_mask.dtype is not torch.bool or fit_mask.dtype is not torch.bool:
            raise TypeError("source_mask and fit_mask must have dtype torch.bool")
        if fit_mask.shape != source_mask.shape:
            raise ValueError("fit_mask must have source_mask shape")
        if not 0 <= num_dest_segments < dest_batch_ptr.shape[0]:
            raise ValueError("num_dest_segments is outside dest_batch_ptr")
        min_pointer_size = num_dest_segments + num_systems + 2
        if dest_batch_ptr.shape[0] < min_pointer_size:
            fit_mask.zero_()
            return

        lengths = source_batch_ptr[1:] - source_batch_ptr[:-1]
        selected_lengths = torch.where(source_mask, lengths, torch.zeros_like(lengths))
        offsets = torch.cumsum(selected_lengths, dim=0) - selected_lengths
        base = dest_batch_ptr[num_dest_segments]
        fit_mask.copy_(source_mask & (base + offsets + lengths <= dest_capacity))

    def put_masked_segmented(
        self,
        source: torch.Tensor,
        source_batch_ptr: torch.Tensor,
        source_mask: torch.Tensor,
        dest: torch.Tensor,
        dest_batch_ptr: torch.Tensor,
        num_dest_segments: int,
        source_data_copied: torch.Tensor,
    ) -> torch.Tensor | None:
        num_systems = source_batch_ptr.shape[0] - 1
        if num_systems == 0:
            return None
        _check_same_device(
            source,
            source_batch_ptr,
            source_mask,
            dest,
            dest_batch_ptr,
            source_data_copied,
        )
        min_pointer_size = num_dest_segments + num_systems + 2
        if dest_batch_ptr.shape[0] < min_pointer_size:
            raise ValueError(
                f"dest_batch_ptr must have size >= {min_pointer_size}; "
                f"got {dest_batch_ptr.shape[0]}"
            )
        if source_data_copied.shape != source_mask.shape or source_data_copied.dtype is not torch.bool:
            raise ValueError("source_data_copied must be a bool tensor with source_mask shape")
        source_rows, dest_rows = _as_rows(source), _as_rows(dest)
        if source_rows.shape[1] != dest_rows.shape[1]:
            raise ValueError("source and dest must have matching trailing element counts")

        fit_mask = torch.zeros_like(source_mask)
        self.compute_put_fit_mask_segmented(
            source_batch_ptr,
            source_mask,
            dest_batch_ptr,
            num_dest_segments,
            dest_rows.shape[0],
            fit_mask,
        )
        source_data_copied.copy_(fit_mask)
        selected = torch.nonzero(fit_mask, as_tuple=False).flatten()
        new_num_dest = torch.tensor(
            [num_dest_segments + selected.numel()],
            device=source.device,
            dtype=torch.int32,
        )
        if selected.numel() == 0:
            return new_num_dest

        starts = source_batch_ptr[selected]
        lengths = source_batch_ptr[selected + 1] - starts
        total = int(lengths.sum().item())
        if total > 0:
            offsets = torch.cumsum(lengths, dim=0) - lengths
            element_offsets = torch.repeat_interleave(offsets, lengths, output_size=total)
            element_starts = torch.repeat_interleave(starts, lengths, output_size=total)
            local = torch.arange(total, device=source.device, dtype=starts.dtype) - element_offsets
            element_indices = element_starts + local
            base = int(dest_batch_ptr[num_dest_segments].item())
            values = source_rows.index_select(0, element_indices).clone()
            dest_rows[base : base + total].copy_(values)
        else:
            base = int(dest_batch_ptr[num_dest_segments].item())

        ends = base + torch.cumsum(lengths, dim=0)
        dest_batch_ptr[num_dest_segments + 1 : num_dest_segments + 1 + selected.numel()].copy_(ends)
        return new_num_dest

    def defrag_segmented(
        self,
        source: torch.Tensor,
        source_batch_ptr: torch.Tensor,
        source_data_copied: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_device(source, source_batch_ptr, source_data_copied)
        num_systems = source_batch_ptr.shape[0] - 1
        num_kept = torch.zeros(1, device=source.device, dtype=torch.int32)
        if num_systems == 0:
            source_batch_ptr[0] = 0
            return num_kept
        if source_data_copied.dtype is not torch.bool or source_data_copied.shape != (num_systems,):
            raise ValueError("source_data_copied must be a bool tensor with one entry per segment")
        source_rows = _as_rows(source)
        kept_indices = torch.nonzero(~source_data_copied, as_tuple=False).flatten()
        num_kept_value = kept_indices.numel()
        num_kept[0] = num_kept_value
        lengths = source_batch_ptr[kept_indices + 1] - source_batch_ptr[kept_indices]
        total = int(lengths.sum().item())
        if num_kept_value:
            starts = source_batch_ptr[kept_indices]
            offsets = torch.cumsum(lengths, dim=0) - lengths
            element_offsets = torch.repeat_interleave(offsets, lengths, output_size=total)
            element_starts = torch.repeat_interleave(starts, lengths, output_size=total)
            local = torch.arange(total, device=source.device, dtype=starts.dtype) - element_offsets
            element_indices = element_starts + local
            kept = source_rows.index_select(0, element_indices).clone()
        else:
            kept = source_rows[:0].clone()
        source.zero_()
        if total:
            _as_rows(source)[:total].copy_(kept)

        new_ptr = torch.cat(
            [
                torch.zeros(1, device=source.device, dtype=source_batch_ptr.dtype),
                torch.cumsum(lengths, dim=0),
            ]
        )
        source_batch_ptr[: num_kept_value + 1].copy_(new_ptr)
        if num_kept_value + 1 < source_batch_ptr.shape[0]:
            source_batch_ptr[num_kept_value + 1 :].fill_(total)
        return num_kept

    def expand_segments(
        self,
        segment_indices: torch.Tensor,
        batch_ptr: torch.Tensor,
        *,
        index_dtype: torch.dtype,
    ) -> torch.Tensor:
        _check_same_device(segment_indices, batch_ptr)
        if index_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Unsupported index_dtype {index_dtype}; expected torch.int32 or torch.int64"
            )
        starts = batch_ptr[segment_indices].to(index_dtype)
        ends = batch_ptr[segment_indices + 1].to(index_dtype)
        lengths = ends - starts
        total = int(lengths.sum().item())
        if index_dtype is torch.int32 and total > _INT32_MAX:
            raise ValueError(f"Total element count {total} exceeds int32 maximum ({_INT32_MAX})")
        if total == 0:
            return torch.empty(0, device=batch_ptr.device, dtype=index_dtype)
        offsets = torch.cumsum(lengths, dim=0) - lengths
        repeated_starts = torch.repeat_interleave(starts, lengths, output_size=total)
        repeated_offsets = torch.repeat_interleave(offsets, lengths, output_size=total)
        local = torch.arange(total, device=batch_ptr.device, dtype=index_dtype) - repeated_offsets
        return repeated_starts + local


def validate_backend_name(name: str) -> StorageBackendName:
    """Validate a user-facing backend name without importing a backend."""
    allowed = {"auto", "torch", "triton", "hip", "warp"}
    if name not in allowed:
        raise ValueError(f"Unknown storage backend {name!r}; expected one of {sorted(allowed)}")
    return name  # type: ignore[return-value]


__all__ = [
    "StorageBackend",
    "StorageBackendName",
    "TorchStorageBackend",
    "validate_backend_name",
]
