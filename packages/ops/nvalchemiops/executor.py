# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy executor binding for pre-resolved operation selections.

The registry owns capability metadata and selection.  This module owns the
small, operation-neutral step that turns a selected implementation metadata
record into one callable entrypoint.  Operation dispatchers provide the
entrypoint name and their framework-owned legacy handler; this module never
contains an operation dispatch table.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, TYPE_CHECKING

import torch

from nvalchemiops._backend_types import LEGACY_IMPLEMENTATION_ID

if TYPE_CHECKING:
    from nvalchemiops.backend import BackendSelection, ImplementationRegistry

ExecutorCallable = Callable[..., Any]
_ENTRYPOINT_CACHE: dict[tuple[str, str], ExecutorCallable] = {}


def clear_entrypoint_cache() -> None:
    """Clear cached entrypoints for test isolation or supported module reloads.

    Normal production processes do not need invalidation: executor module paths
    are stable for the lifetime of the process.  Tests and development tools
    that replace a module or entrypoint explicitly may use this hook before
    resolving it again.
    """
    _ENTRYPOINT_CACHE.clear()


def _error(
    message: str, *, cause: BaseException | None = None
) -> "BackendUnavailableError":
    from nvalchemiops.backend import BackendUnavailableError

    error = BackendUnavailableError(message)
    if cause is not None:
        error.__cause__ = cause
    return error


def _owner_context(owner: str) -> str:
    """Format executor ownership for actionable package-boundary errors."""
    return f"executor_owner={owner!r} ({owner}-owned package)"


def _load_registered_entrypoint(
    implementation_id: str,
    operation: str,
    family: str,
    strategy: str | None,
    entrypoint_name: str,
    *,
    registry: ImplementationRegistry,
) -> ExecutorCallable:
    """Load an entrypoint using scalar selection metadata only."""
    implementation = registry.get(implementation_id)
    if implementation is None:
        raise _error(
            "executor selection references an unknown implementation: "
            f"{implementation_id!r}"
        )

    operation_matches = implementation.operation in {operation, "*"}
    if not operation_matches:
        raise _error(
            "executor selection operation mismatch: "
            f"implementation {implementation_id!r} declares "
            f"{implementation.operation!r}, selection targets {operation!r}"
        )
    if implementation.family != family or implementation.strategy != strategy:
        raise _error(
            "executor selection metadata mismatch for "
            f"implementation {implementation_id!r}: "
            f"catalog family/strategy=({implementation.family!r}, "
            f"{implementation.strategy!r}), selection=({family!r}, "
            f"{strategy!r})"
        )

    owner = implementation.executor_owner or "unknown"
    if implementation_id == LEGACY_IMPLEMENTATION_ID:
        raise _error(
            "legacy implementation is framework-owned and has no registry "
            f"executor: implementation={implementation_id!r}, "
            f"operation={operation!r}, {_owner_context(owner)}"
        )
    if implementation.executor is None:
        raise _error(
            "registered implementation has no executor: "
            f"implementation={implementation_id!r}, operation={operation!r}, "
            f"{_owner_context(owner)}"
        )
    if entrypoint_name not in implementation.entrypoints:
        raise _error(
            "entrypoint is not declared by selected implementation: "
            f"implementation={implementation_id!r}, operation={operation!r}, "
            f"{_owner_context(owner)}, entrypoint={entrypoint_name!r}"
        )
    try:
        return _load_cached(implementation.executor, entrypoint_name)
    except RuntimeError as exc:
        message = str(exc)
        if _owner_context(owner) not in message:
            message = (
                f"{message}; implementation={implementation_id!r}, "
                f"operation={operation!r}, {_owner_context(owner)}"
            )
        raise type(exc)(message) from exc


def _load_cached(executor: str, entrypoint_name: str) -> ExecutorCallable:
    """Import and validate one executor module entrypoint exactly once."""
    cache_key = (executor, entrypoint_name)
    cached = _ENTRYPOINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        module = importlib.import_module(executor)
    except Exception as exc:  # module import errors need selection context
        raise _error(
            "failed to import executor "
            f"{executor!r} for entrypoint {entrypoint_name!r}",
            cause=exc,
        ) from exc

    try:
        entrypoint = getattr(module, entrypoint_name)
    except AttributeError as exc:
        raise _error(
            "executor module is missing entrypoint "
            f"{entrypoint_name!r}: module={executor!r}",
            cause=exc,
        ) from exc
    if not callable(entrypoint):
        raise _error(
            "executor entrypoint is not callable: "
            f"module={executor!r}, entrypoint={entrypoint_name!r}"
        )
    _ENTRYPOINT_CACHE[cache_key] = entrypoint
    return entrypoint


def load_entrypoint(
    selection: BackendSelection,
    entrypoint_name: str,
    *,
    registry: ImplementationRegistry | None = None,
) -> ExecutorCallable:
    """Load one callable declared by a pre-resolved implementation.

    This function intentionally accepts a registry as data rather than being
    a registry method.  It keeps capability resolution and executor loading
    separate and makes loader behavior independently mockable in tests.
    """
    if not entrypoint_name or not entrypoint_name.isidentifier():
        raise ValueError("entrypoint_name must be a valid non-empty identifier")
    if registry is None:
        from nvalchemiops.backend import DEFAULT_IMPLEMENTATION_REGISTRY

        registry = DEFAULT_IMPLEMENTATION_REGISTRY

    return _load_registered_entrypoint(
        selection.implementation_id,
        selection.operation,
        selection.family,
        selection.strategy,
        entrypoint_name,
        registry=registry,
    )


@torch.compiler.allow_in_graph
def _execute_loaded(
    implementation_id: str,
    operation: str,
    family: str,
    strategy: str | None,
    entrypoint_name: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a loaded entrypoint through a compile-safe scalar boundary."""
    from nvalchemiops.backend import DEFAULT_IMPLEMENTATION_REGISTRY

    entrypoint = _load_registered_entrypoint(
        implementation_id,
        operation,
        family,
        strategy,
        entrypoint_name,
        registry=DEFAULT_IMPLEMENTATION_REGISTRY,
    )
    return entrypoint(*args, **kwargs)


def execute_selected(
    selection: BackendSelection,
    entrypoint_name: str,
    legacy_fn: ExecutorCallable,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a selected entrypoint with one explicit legacy boundary.

    The compile-time path passes only scalar selection metadata through
    ``allow_in_graph``.  This keeps lazy module import and callable lookup out
    of the traced graph while preserving the same generic binding rules.
    """
    if selection.implementation_id == LEGACY_IMPLEMENTATION_ID:
        return legacy_fn(*args, **kwargs)
    if torch.compiler.is_compiling():
        return _execute_loaded(
            selection.implementation_id,
            selection.operation,
            selection.family,
            selection.strategy,
            entrypoint_name,
            *args,
            **kwargs,
        )
    return load_entrypoint(selection, entrypoint_name)(*args, **kwargs)


__all__ = ["clear_entrypoint_cache", "execute_selected", "load_entrypoint"]
