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
from functools import lru_cache
from typing import Any, Callable, TYPE_CHECKING

from nvalchemiops._backend_types import LEGACY_IMPLEMENTATION_ID

if TYPE_CHECKING:
    from nvalchemiops.backend import BackendSelection, ImplementationRegistry

ExecutorCallable = Callable[..., Any]


def _error(message: str, *, cause: BaseException | None = None) -> RuntimeError:
    from nvalchemiops.backend import BackendUnavailableError

    error = BackendUnavailableError(message)
    if cause is not None:
        error.__cause__ = cause
    return error


def _validate_selection_metadata(
    selection: BackendSelection,
    implementation: Any,
) -> None:
    """Reject a selection that no longer matches the registry metadata."""
    operation_matches = implementation.operation in {selection.operation, "*"}
    if not operation_matches:
        raise _error(
            "executor selection operation mismatch: "
            f"implementation {selection.implementation_id!r} declares "
            f"{implementation.operation!r}, selection targets "
            f"{selection.operation!r}"
        )
    if (
        implementation.family != selection.family
        or implementation.strategy != selection.strategy
    ):
        raise _error(
            "executor selection metadata mismatch for "
            f"implementation {selection.implementation_id!r}: "
            f"catalog family/strategy=({implementation.family!r}, "
            f"{implementation.strategy!r}), selection="
            f"({selection.family!r}, {selection.strategy!r})"
        )


@lru_cache(maxsize=None)
def _load_cached(executor: str, entrypoint_name: str) -> ExecutorCallable:
    """Import and validate one executor module entrypoint exactly once."""
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

    implementation = registry.get(selection.implementation_id)
    if implementation is None:
        raise _error(
            "executor selection references an unknown implementation: "
            f"{selection.implementation_id!r}"
        )
    _validate_selection_metadata(selection, implementation)
    owner = implementation.executor_owner or "unknown"
    if implementation.implementation_id == LEGACY_IMPLEMENTATION_ID:
        raise _error(
            "legacy implementation is framework-owned and has no registry "
            f"executor: implementation={selection.implementation_id!r}, "
            f"operation={selection.operation!r}, executor_owner={owner!r}"
        )
    if implementation.executor is None:
        raise _error(
            "registered implementation has no executor: "
            f"implementation={selection.implementation_id!r}, "
            f"operation={selection.operation!r}, executor_owner={owner!r}"
        )
    if entrypoint_name not in implementation.entrypoints:
        raise _error(
            "entrypoint is not declared by selected implementation: "
            f"implementation={selection.implementation_id!r}, "
            f"operation={selection.operation!r}, "
            f"executor_owner={owner!r}, entrypoint={entrypoint_name!r}"
        )
    try:
        return _load_cached(implementation.executor, entrypoint_name)
    except RuntimeError as exc:
        message = str(exc)
        if f"executor_owner={owner!r}" not in message:
            message = (
                f"{message}; implementation={selection.implementation_id!r}, "
                f"operation={selection.operation!r}, "
                f"executor_owner={owner!r}"
            )
        raise type(exc)(message) from exc


def execute_selected(
    selection: BackendSelection,
    entrypoint_name: str,
    legacy_fn: ExecutorCallable,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a selected entrypoint with one explicit legacy boundary."""
    if selection.implementation_id == LEGACY_IMPLEMENTATION_ID:
        return legacy_fn(*args, **kwargs)
    return load_entrypoint(selection, entrypoint_name)(*args, **kwargs)


__all__ = ["execute_selected", "load_entrypoint"]
