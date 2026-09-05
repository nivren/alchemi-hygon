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


def validate_backend_name(name: str) -> StorageBackendName:
    """Validate a user-facing backend name without importing a backend."""
    allowed = {"auto", "torch", "triton", "hip", "warp"}
    if name not in allowed:
        raise ValueError(f"Unknown storage backend {name!r}; expected one of {sorted(allowed)}")
    return name  # type: ignore[return-value]


__all__ = ["StorageBackend", "StorageBackendName", "validate_backend_name"]
