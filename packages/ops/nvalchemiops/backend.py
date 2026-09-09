# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-aware backend selection shared by device-neutral dispatchers.

The registry deliberately selects implementations without importing a backend.
In particular, ``warp`` is a legacy framework-owned execution path: resolving
it records the choice, but does not import or probe Warp.  This lets HCU
reference imports remain Warp-free while preserving the upstream default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal

BackendName = Literal[
    "auto",
    "torch_reference",
    "torch_reference_cell_list",
    "triton",
    "hip",
    "warp",
]
_KNOWN_BACKENDS = (
    "auto",
    "torch_reference",
    "torch_reference_cell_list",
    "triton",
    "hip",
    "warp",
)
_AUTO_PRIORITY = ("triton", "hip", "torch_reference")


class BackendUnavailableError(RuntimeError):
    """Raised when no registered backend can honour a request."""


class BackendAutoSelectionWarning(UserWarning):
    """Emitted once when ``auto`` selects a verified non-legacy backend."""


@dataclass(frozen=True)
class BackendCapability:
    """A verified implementation width for one operation."""

    backend: str
    operation: str
    features: frozenset[str]
    dtypes: frozenset[str] = frozenset({"float32", "float64"})
    max_gradient_order: int = 0
    devices: frozenset[str] = frozenset({"cpu", "cuda"})
    excluded_feature_sets: tuple[frozenset[str], ...] = ()
    evidence: str = ""

    def supports(
        self,
        *,
        device: str,
        dtype: str | None,
        gradient_order: int,
        features: frozenset[str],
    ) -> bool:
        return (
            (device == "unspecified" or device in self.devices)
            and (dtype is None or dtype in self.dtypes)
            and gradient_order <= self.max_gradient_order
            and features.issubset(self.features)
            and not any(excluded.issubset(features) for excluded in self.excluded_feature_sets)
        )


@dataclass(frozen=True)
class BackendSelection:
    """Auditable result of a backend selection decision."""

    requested: str
    selected: str
    operation: str
    device: str
    dtype: str | None
    gradient_order: int
    features: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, str | int | list[str] | None]:
        """Return a serialization-friendly record for logs and probe reports."""
        return {
            "requested": self.requested,
            "selected": self.selected,
            "operation": self.operation,
            "device": self.device,
            "dtype": self.dtype,
            "gradient_order": self.gradient_order,
            "features": list(self.features),
            "reason": self.reason,
        }


def _device_label(device: Any) -> str:
    if device is None:
        return "unspecified"
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type)
    return str(device).split(":", maxsplit=1)[0]


def _dtype_label(dtype: Any) -> str | None:
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _feature_set(features: Iterable[str] | None) -> frozenset[str]:
    return frozenset(features or ())


_CAPABILITIES: tuple[BackendCapability, ...] = (
    BackendCapability(
        backend="torch_reference",
        operation="neighbor_list",
        features=frozenset(
            {
                "no_pbc",
                "periodic",
                "full",
                "half",
                "matrix",
                "coo",
                "distances",
                "vectors",
            }
        ),
        excluded_feature_sets=(frozenset({"periodic", "half"}),),
        evidence="G1/G2 Torch-reference neighbor contracts",
    ),
    BackendCapability(
        backend="torch_reference_cell_list",
        operation="neighbor_list",
        features=frozenset(
            {
                "no_pbc",
                "full",
                "half",
                "matrix",
                "coo",
                "distances",
                "vectors",
            }
        ),
        devices=frozenset({"cpu", "cuda"}),
        evidence="G2 opt-in no-PBC Torch reference cell-list contract",
    ),
    BackendCapability(
        backend="torch_reference",
        operation="lj_energy_forces",
        features=frozenset({"no_pbc", "periodic", "full", "half", "forces"}),
        excluded_feature_sets=(frozenset({"periodic", "half"}),),
        max_gradient_order=2,
        evidence="G1 Torch-reference LJ force/curvature contracts",
    ),
    BackendCapability(
        backend="torch_reference",
        operation="velocity_verlet",
        features=frozenset({"fixed_cell"}),
        max_gradient_order=1,
        evidence="G2 velocity-Verlet reference contracts",
    ),
    BackendCapability(
        backend="torch_reference",
        operation="fire",
        features=frozenset({"fixed_cell"}),
        max_gradient_order=1,
        evidence="G2 FIRE/FIRE2 reference contracts",
    ),
    BackendCapability(
        backend="torch_reference",
        operation="kinetics",
        features=frozenset({"per_graph"}),
        max_gradient_order=1,
        evidence="G2 kinetic-energy/temperature reference contracts",
    ),
    BackendCapability(
        backend="torch_reference",
        operation="periodic_wrap",
        features=frozenset({"inplace", "periodic"}),
        evidence="G2 periodic-hook reference contracts",
    ),
    BackendCapability(
        backend="torch_reference",
        operation="segmented_reduce",
        features=frozenset({"per_graph"}),
        evidence="G2 observer reference contracts",
    ),
)

_AUTO_WARNED: set[tuple[str, str, str | None, int, tuple[str, ...]]] = set()


def backend_capabilities(*, operation: str | None = None) -> tuple[BackendCapability, ...]:
    """Return registered capabilities without importing implementations."""
    if operation is None:
        return _CAPABILITIES
    return tuple(capability for capability in _CAPABILITIES if capability.operation == operation)


def validate_backend_name(requested: str | None) -> None:
    """Validate a public backend spelling without selecting an implementation."""
    if requested is not None and requested not in _KNOWN_BACKENDS:
        choices = ", ".join(_KNOWN_BACKENDS)
        raise ValueError(f"unknown backend {requested!r}; choices are: {choices}")


def _matching_capabilities(
    backend: str,
    *,
    operation: str,
    device: str,
    dtype: str | None,
    gradient_order: int,
    features: frozenset[str],
) -> tuple[BackendCapability, ...]:
    return tuple(
        capability
        for capability in _CAPABILITIES
        if capability.backend == backend
        and capability.operation == operation
        and capability.supports(
            device=device,
            dtype=dtype,
            gradient_order=gradient_order,
            features=features,
        )
    )


def resolve_backend(
    requested: str | None = "torch_reference",
    *,
    operation: str,
    device: Any = None,
    dtype: Any = None,
    gradient_order: int = 0,
    features: Iterable[str] | None = None,
) -> BackendSelection:
    """Resolve a request against verified operation capabilities.

    ``None`` and ``"warp"`` preserve the framework's legacy Warp default.
    They are intentionally not capability-probed here. Direct Torch
    dispatchers reject such a selection because they do not own Warp's
    executor.
    """
    validate_backend_name(requested)
    if gradient_order < 0:
        raise ValueError("gradient_order must be non-negative")

    effective_requested = "warp" if requested is None else requested
    device_label = _device_label(device)
    dtype_label = _dtype_label(dtype)
    feature_set = _feature_set(features)
    feature_labels = tuple(sorted(feature_set))

    if effective_requested == "warp":
        return BackendSelection(
            requested="warp" if requested is None else effective_requested,
            selected="warp",
            operation=operation,
            device=device_label,
            dtype=dtype_label,
            gradient_order=gradient_order,
            features=feature_labels,
            reason="legacy upstream Warp default; execution remains framework-owned",
        )

    candidates = _AUTO_PRIORITY if effective_requested == "auto" else (effective_requested,)
    for candidate in candidates:
        matches = _matching_capabilities(
            candidate,
            operation=operation,
            device=device_label,
            dtype=dtype_label,
            gradient_order=gradient_order,
            features=feature_set,
        )
        if not matches:
            continue
        reason = (
            "explicit verified backend capability"
            if effective_requested != "auto"
            else "auto selected the highest-priority verified capability"
        )
        selection = BackendSelection(
            requested=effective_requested,
            selected=candidate,
            operation=operation,
            device=device_label,
            dtype=dtype_label,
            gradient_order=gradient_order,
            features=feature_labels,
            reason=reason,
        )
        if effective_requested == "auto":
            warning_key = (operation, device_label, dtype_label, gradient_order, feature_labels)
            if warning_key not in _AUTO_WARNED:
                _AUTO_WARNED.add(warning_key)
                warnings.warn(
                    "backend='auto' selected "
                    f"{candidate!r} for {operation!r}: {reason}",
                    BackendAutoSelectionWarning,
                    stacklevel=2,
                )
        return selection

    feature_text = ", ".join(feature_labels) or "none"
    raise BackendUnavailableError(
        f"backend {effective_requested!r} has no verified capability for {operation!r} "
        f"(device={device_label}, dtype={dtype_label}, gradient_order={gradient_order}, "
        f"features={feature_text})"
    )


__all__ = [
    "BackendAutoSelectionWarning",
    "BackendCapability",
    "BackendName",
    "BackendSelection",
    "BackendUnavailableError",
    "backend_capabilities",
    "resolve_backend",
    "validate_backend_name",
]
