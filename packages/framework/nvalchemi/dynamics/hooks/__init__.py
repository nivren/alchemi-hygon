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
"""
Dynamics hooks for observation, safety, and behavior modification.

This sub-package provides concrete hook implementations that plug into the
:class:`~nvalchemi.dynamics.base.BaseDynamics` hook system.  Every class
satisfies the :class:`~nvalchemi.hooks.Hook` protocol and can be
registered with any dynamics engine via
:meth:`~nvalchemi.dynamics.base.BaseDynamics.register_hook`.

Hooks are organized into the following modules:

.. list-table::
   :widths: 20 80

   * - :mod:`snapshot`
     - Save batch state to a :class:`~nvalchemi.dynamics.sinks.DataSink`.
   * - :mod:`logging`
     - Log scalar observables (energy, temperature, fmax, etc.).
   * - :mod:`safety`
     - Numerical safety guards (NaN detection, force clamping).
   * - :mod:`monitors`
     - Long-running diagnostic monitors (energy drift).
   * - :mod:`freeze`
     - Freeze selected atoms by category during dynamics.
   * - :mod:`cell_align`
     - Align periodic cells to upper-triangular form for variable-cell optimization.
   * - :mod:`nvalchemi.hooks.physicsnemo_profiling`
     - PyTorch profiler trace capture through PhysicsNeMo.

All hooks implement the :class:`~nvalchemi.hooks.Hook` protocol and accept
a :class:`~nvalchemi.hooks.DynamicsContext` plus a stage enum in their
``__call__`` method.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AlignCellHook": ("nvalchemi.dynamics.hooks.cell_align", "AlignCellHook"),
    "FreezeAtomsHook": ("nvalchemi.dynamics.hooks.freeze", "FreezeAtomsHook"),
    "LoggingHook": ("nvalchemi.dynamics.hooks.logging", "LoggingHook"),
    "EnergyDriftMonitorHook": (
        "nvalchemi.dynamics.hooks.monitors",
        "EnergyDriftMonitorHook",
    ),
    "MaxForceClampHook": ("nvalchemi.dynamics.hooks.safety", "MaxForceClampHook"),
    "NaNDetectorHook": ("nvalchemi.dynamics.hooks.safety", "NaNDetectorHook"),
    "ConvergedSnapshotHook": (
        "nvalchemi.dynamics.hooks.snapshot",
        "ConvergedSnapshotHook",
    ),
    "SnapshotHook": ("nvalchemi.dynamics.hooks.snapshot", "SnapshotHook"),
    "StageTimingHook": ("nvalchemi.hooks.stage_timing", "StageTimingHook"),
}

__all__ = [
    "AlignCellHook",
    "ConvergedSnapshotHook",
    "EnergyDriftMonitorHook",
    "FreezeAtomsHook",
    "LoggingHook",
    "MaxForceClampHook",
    "NaNDetectorHook",
    "SnapshotHook",
    "StageTimingHook",
    "TorchProfilerHook",
]

_REMOVED_PROFILER_HOOKS = {"ProfilerHook"}


def __getattr__(name: str) -> object:
    """Load optional profiler support or reject removed hook names."""
    if name in _EXPORTS:
        module_name, symbol = _EXPORTS[name]
        value = getattr(import_module(module_name), symbol)
        globals()[name] = value
        return value
    if name == "TorchProfilerHook":
        from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook

        return TorchProfilerHook
    if name in _REMOVED_PROFILER_HOOKS:
        raise ImportError(
            f"nvalchemi.dynamics.hooks.{name} was removed. "
            "Use nvalchemi.dynamics.hooks.TorchProfilerHook for PyTorch traces or nvalchemi.dynamics.hooks.StageTimingHook for per-stage timing instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
