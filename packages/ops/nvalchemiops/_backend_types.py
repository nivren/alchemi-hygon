# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free value objects shared by registry metadata modules."""

from __future__ import annotations

from dataclasses import dataclass

BackendFamily = str
ImplementationId = str
LEGACY_IMPLEMENTATION_ID = "warp.legacy-upstream-v1"


@dataclass(frozen=True)
class Implementation:
    """One registered operation implementation and its verified capability width."""

    implementation_id: ImplementationId
    operation: str
    family: BackendFamily
    strategy: str | None = None
    executor: str | None = None
    features: frozenset[str] = frozenset()
    dtypes: frozenset[str] = frozenset({"float32", "float64"})
    max_gradient_order: int = 0
    devices: frozenset[str] = frozenset({"cpu", "cuda"})
    excluded_feature_sets: tuple[frozenset[str], ...] = ()
    evidence: str = ""
    default_strategy: bool = False

    def supports(
        self,
        *,
        device: str,
        dtype: str | None,
        gradient_order: int,
        features: frozenset[str],
    ) -> bool:
        """Return whether this implementation can honour the request contract."""
        return (
            (device == "unspecified" or device in self.devices)
            and (dtype is None or dtype in self.dtypes)
            and gradient_order <= self.max_gradient_order
            and features.issubset(self.features)
            and not any(excluded.issubset(features) for excluded in self.excluded_feature_sets)
        )

    @property
    def backend(self) -> str:
        """Compatibility view for capability inventory callers."""
        return self.family
