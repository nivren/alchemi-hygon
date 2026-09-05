# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend capability and selection records shared by device-neutral dispatchers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BackendName = Literal["auto", "torch_reference", "triton", "hip", "warp"]
_KNOWN_BACKENDS = ("auto", "torch_reference", "triton", "hip", "warp")


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend is not registered for an operation."""


@dataclass(frozen=True)
class BackendSelection:
    """Auditable result of a backend selection decision."""

    requested: str
    selected: str
    operation: str
    device: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """Return a serialization-friendly record for logs and probe reports."""
        return {
            "requested": self.requested,
            "selected": self.selected,
            "operation": self.operation,
            "device": self.device,
            "reason": self.reason,
        }


def _device_label(device: Any) -> str:
    if device is None:
        return "unspecified"
    return str(device)


def resolve_backend(
    requested: str = "torch_reference",
    *,
    operation: str,
    device: Any = None,
) -> BackendSelection:
    """Resolve a backend without importing Warp or probing by name alone.

    At this stage only the Torch reference implementation is registered.  An
    explicit ``auto`` request therefore selects it and records that reason;
    requesting an unregistered optimized backend fails instead of silently
    switching implementation or device.
    """
    if requested not in _KNOWN_BACKENDS:
        choices = ", ".join(_KNOWN_BACKENDS)
        raise ValueError(f"unknown backend {requested!r}; choices are: {choices}")
    if requested in ("auto", "torch_reference"):
        reason = (
            "auto selected the only registered backend in this slice"
            if requested == "auto"
            else "explicit Torch reference backend"
        )
        return BackendSelection(
            requested=requested,
            selected="torch_reference",
            operation=operation,
            device=_device_label(device),
            reason=reason,
        )
    raise BackendUnavailableError(
        f"backend {requested!r} is not registered for {operation!r}; "
        "use backend='torch_reference' or implement and validate this backend first"
    )


__all__ = [
    "BackendName",
    "BackendSelection",
    "BackendUnavailableError",
    "resolve_backend",
]
