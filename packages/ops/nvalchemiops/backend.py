# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability registry for device-neutral operation dispatch.

The registry answers only whether a concrete implementation can satisfy an
operation contract. Platform policy and whole-pipeline planning are separate
M2/M3 concerns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, TypeAlias

from nvalchemiops._backend_types import (
    LEGACY_IMPLEMENTATION_ID as _LEGACY_IMPLEMENTATION_ID,
)
from nvalchemiops._backend_types import Implementation

BackendRequest: TypeAlias = str | None
BackendFamily: TypeAlias = str
ImplementationId: TypeAlias = str

_AUTO_FAMILIES = ("triton", "hip", "torch_reference")


class BackendUnavailableError(RuntimeError):
    """Raised when a request has no registered implementation for its contract."""


class BackendAutoSelectionWarning(UserWarning):
    """Emitted once when an explicit ``auto`` request selects an implementation."""


@dataclass(frozen=True)
class BackendSelection:
    """An auditable, pre-resolved implementation decision."""

    requested: BackendRequest
    implementation_id: ImplementationId
    family: BackendFamily
    strategy: str | None
    operation: str
    device: str
    dtype: str | None
    gradient_order: int
    features: tuple[str, ...]
    reason: str
    profile_id: str | None = None

    @property
    def selected(self) -> str:
        """Compatibility view for family-based legacy call sites."""
        return self.family

    def as_dict(self) -> dict[str, str | int | list[str] | None]:
        """Return a serialization-friendly decision record."""
        return {
            "requested": self.requested,
            "implementation_id": self.implementation_id,
            "family": self.family,
            "strategy": self.strategy,
            "selected": self.family,
            "operation": self.operation,
            "device": self.device,
            "dtype": self.dtype,
            "gradient_order": self.gradient_order,
            "features": list(self.features),
            "reason": self.reason,
            "profile_id": self.profile_id,
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


def validate_backend_request(requested: BackendRequest) -> None:
    """Validate request shape; registry membership is resolved centrally."""
    if requested is not None and (not isinstance(requested, str) or not requested):
        raise ValueError("backend request must be a non-empty string or None")


# Framework constructors still use the historical helper name.
validate_backend_name = validate_backend_request


class ImplementationRegistry:
    """Ordered registry of lazy implementation metadata."""

    def __init__(self, implementations: Iterable[Implementation] = ()) -> None:
        self._implementations: list[Implementation] = []
        self._by_id: dict[ImplementationId, Implementation] = {}
        for implementation in implementations:
            self.register(implementation)

    def register(self, implementation: Implementation) -> None:
        """Register an implementation, rejecting duplicate stable identifiers."""
        if implementation.implementation_id in self._by_id:
            raise ValueError(
                "duplicate implementation_id "
                f"{implementation.implementation_id!r}"
            )
        if not implementation.operation:
            raise ValueError("implementation operation must be non-empty")
        if not implementation.family:
            raise ValueError("implementation family must be non-empty")
        if implementation.implementation_id == _LEGACY_IMPLEMENTATION_ID:
            if (
                implementation.operation != "*"
                or implementation.family != "warp"
                or implementation.executor is not None
                or implementation.entrypoints
                or implementation.executor_owner != "framework"
            ):
                raise ValueError(
                    "the legacy implementation must be a framework-owned "
                    "wildcard without a registry executor"
                )
        else:
            if not implementation.executor:
                raise ValueError(
                    "non-legacy implementation must declare an executor module"
                )
            if implementation.executor_owner not in {"ops", "framework"}:
                raise ValueError(
                    "non-legacy implementation must declare executor_owner "
                    "as 'ops' or 'framework'"
                )
            if not implementation.entrypoints or len(
                set(implementation.entrypoints)
            ) != len(implementation.entrypoints):
                raise ValueError(
                    "non-legacy implementation must declare unique entrypoints"
                )
            if any(
                not entrypoint or not entrypoint.isidentifier()
                for entrypoint in implementation.entrypoints
            ):
                raise ValueError(
                    "implementation entrypoints must be valid identifiers"
                )
            if any(
                not part or not part.isidentifier()
                for part in implementation.executor.split(".")
            ):
                raise ValueError(
                    "implementation executor must be a dotted module path"
                )
        self._by_id[implementation.implementation_id] = implementation
        self._implementations.append(implementation)

    def get(self, implementation_id: ImplementationId) -> Implementation | None:
        """Return metadata for one implementation without importing it."""
        return self._by_id.get(implementation_id)

    def implementations(self, *, operation: str | None = None) -> tuple[Implementation, ...]:
        """Return registered metadata without importing any executor."""
        if operation is None:
            return tuple(self._implementations)
        return tuple(
            implementation
            for implementation in self._implementations
            if implementation.operation in {operation, "*"}
        )

    def _legacy_selection(
        self,
        *,
        requested: BackendRequest,
        operation: str,
        device: str,
        dtype: str | None,
        gradient_order: int,
        features: tuple[str, ...],
        strategy: str | None,
    ) -> BackendSelection:
        if strategy is not None:
            raise BackendUnavailableError(
                "legacy Warp request does not resolve an operation strategy; "
                "use an explicit registered backend family"
            )
        legacy = self._by_id[_LEGACY_IMPLEMENTATION_ID]
        return BackendSelection(
            requested=requested,
            implementation_id=legacy.implementation_id,
            family=legacy.family,
            strategy=None,
            operation=operation,
            device=device,
            dtype=dtype,
            gradient_order=gradient_order,
            features=features,
            reason="legacy upstream Warp default; execution remains framework-owned",
        )

    def _matching(
        self,
        candidates: Iterable[Implementation],
        *,
        operation: str,
        strategy: str | None,
        device: str,
        dtype: str | None,
        gradient_order: int,
        features: frozenset[str],
    ) -> tuple[Implementation, ...]:
        return tuple(
            implementation
            for implementation in candidates
            if implementation.operation == operation
            and (
                (strategy is None and implementation.default_strategy)
                or strategy == implementation.strategy
            )
            and implementation.supports(
                device=device,
                dtype=dtype,
                gradient_order=gradient_order,
                features=features,
            )
        )

    def resolve(
        self,
        requested: BackendRequest,
        *,
        operation: str,
        device: Any = None,
        dtype: Any = None,
        gradient_order: int = 0,
        features: Iterable[str] | None = None,
        strategy: str | None = None,
    ) -> BackendSelection:
        """Resolve one operation request without importing an executor."""
        validate_backend_request(requested)
        if gradient_order < 0:
            raise ValueError("gradient_order must be non-negative")
        if strategy is not None and not strategy:
            raise ValueError("strategy must be a non-empty string or None")

        device_label = _device_label(device)
        dtype_label = _dtype_label(dtype)
        feature_set = _feature_set(features)
        feature_labels = tuple(sorted(feature_set))

        if requested in (None, "warp", _LEGACY_IMPLEMENTATION_ID):
            return self._legacy_selection(
                requested=requested,
                operation=operation,
                device=device_label,
                dtype=dtype_label,
                gradient_order=gradient_order,
                features=feature_labels,
                strategy=strategy,
            )

        if requested == "auto" and strategy is not None:
            raise BackendUnavailableError(
                "backend='auto' does not select an operation strategy without a "
                "BackendProfile; request an implementation family explicitly"
            )

        if requested == "auto":
            families = _AUTO_FAMILIES
            request_candidates: tuple[Implementation, ...] = ()
            effective_strategy = strategy
        elif requested in self._by_id:
            families = ("__exact_implementation__",)
            request_candidates = (self._by_id[requested],)
            effective_strategy = (
                request_candidates[0].strategy if strategy is None else strategy
            )
        else:
            families = (requested,)
            request_candidates = ()
            effective_strategy = strategy

        for family in families:
            candidates: Iterable[Implementation]
            if request_candidates:
                candidates = request_candidates
            else:
                candidates = (
                    implementation
                    for implementation in self._implementations
                    if implementation.family == family
                )
            matches = self._matching(
                candidates,
                operation=operation,
                strategy=effective_strategy,
                device=device_label,
                dtype=dtype_label,
                gradient_order=gradient_order,
                features=feature_set,
            )
            if not matches:
                continue
            implementation = matches[0]
            is_auto = requested == "auto"
            reason = (
                "auto selected the highest-priority verified capability"
                if is_auto
                else "explicit verified implementation capability"
            )
            selection = BackendSelection(
                requested=requested,
                implementation_id=implementation.implementation_id,
                family=implementation.family,
                strategy=implementation.strategy,
                operation=operation,
                device=device_label,
                dtype=dtype_label,
                gradient_order=gradient_order,
                features=feature_labels,
                reason=reason,
            )
            if is_auto:
                warning_key = (
                    operation,
                    device_label,
                    dtype_label,
                    gradient_order,
                    feature_labels,
                )
                if warning_key not in _AUTO_WARNED:
                    _AUTO_WARNED.add(warning_key)
                    warnings.warn(
                        "backend='auto' selected "
                        f"{implementation.implementation_id!r} for {operation!r}: {reason}",
                        BackendAutoSelectionWarning,
                        stacklevel=3,
                    )
            return selection

        feature_text = ", ".join(feature_labels) or "none"
        known = {
            "auto",
            "warp",
            *_AUTO_FAMILIES,
            *self._by_id,
            *(item.family for item in self._implementations),
        }
        request_text = (
            f"unknown backend request {requested!r}"
            if requested not in known
            else f"backend request {requested!r} has no verified capability"
        )
        raise BackendUnavailableError(
            f"{request_text} for {operation!r} (device={device_label}, "
            f"dtype={dtype_label}, gradient_order={gradient_order}, "
            f"strategy={strategy!r}, features={feature_text})"
        )


def _default_implementations() -> tuple[Implementation, ...]:
    """Load the ordered metadata catalog without importing any executor."""
    # The catalog's dependency-free value objects and executor path strings
    # keep registry initialization separate from executor imports.
    from nvalchemiops._backend_catalog import default_implementations

    return default_implementations()


DEFAULT_IMPLEMENTATION_REGISTRY = ImplementationRegistry(_default_implementations())
_AUTO_WARNED: set[tuple[str, str, str | None, int, tuple[str, ...]]] = set()


def backend_capabilities(*, operation: str | None = None) -> tuple[Implementation, ...]:
    """Return implementation metadata for the requested operation."""
    return DEFAULT_IMPLEMENTATION_REGISTRY.implementations(operation=operation)


def resolve_backend(
    requested: BackendRequest = "torch_reference",
    *,
    operation: str,
    device: Any = None,
    dtype: Any = None,
    gradient_order: int = 0,
    features: Iterable[str] | None = None,
    strategy: str | None = None,
) -> BackendSelection:
    """Resolve one operation through the default implementation registry."""
    return DEFAULT_IMPLEMENTATION_REGISTRY.resolve(
        requested,
        operation=operation,
        device=device,
        dtype=dtype,
        gradient_order=gradient_order,
        features=features,
        strategy=strategy,
    )


__all__ = [
    "BackendAutoSelectionWarning",
    "BackendFamily",
    "BackendRequest",
    "BackendSelection",
    "BackendUnavailableError",
    "DEFAULT_IMPLEMENTATION_REGISTRY",
    "Implementation",
    "ImplementationId",
    "ImplementationRegistry",
    "backend_capabilities",
    "resolve_backend",
    "validate_backend_name",
    "validate_backend_request",
]
