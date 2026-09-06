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
"""Dynamics simulation framework for molecular systems.

The package exports the locked upstream names through lazy attribute loading.
Importing the namespace itself must remain Warp-free so Torch reference
backends can be selected explicitly on systems where Warp is unavailable.
Requesting a legacy dynamics class still imports its original module and thus
retains the original optional dependency boundary.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BaseDynamics": ("nvalchemi.dynamics.base", "BaseDynamics"),
    "ConvergenceHook": ("nvalchemi.dynamics.base", "ConvergenceHook"),
    "DistributedPipeline": ("nvalchemi.dynamics.base", "DistributedPipeline"),
    "DynamicsStage": ("nvalchemi.dynamics.base", "DynamicsStage"),
    "FusedStage": ("nvalchemi.dynamics.base", "FusedStage"),
    "Hook": ("nvalchemi.dynamics.base", "Hook"),
    "DemoDynamics": ("nvalchemi.dynamics.demo", "DemoDynamics"),
    "NPH": ("nvalchemi.dynamics.integrators", "NPH"),
    "NPT": ("nvalchemi.dynamics.integrators", "NPT"),
    "NVE": ("nvalchemi.dynamics.integrators", "NVE"),
    "NVTLangevin": ("nvalchemi.dynamics.integrators", "NVTLangevin"),
    "NVTNoseHoover": ("nvalchemi.dynamics.integrators", "NVTNoseHoover"),
    "FIRE": ("nvalchemi.dynamics.optimizers", "FIRE"),
    "FIRE2": ("nvalchemi.dynamics.optimizers", "FIRE2"),
    "FIRE2VariableCell": ("nvalchemi.dynamics.optimizers", "FIRE2VariableCell"),
    "FIREVariableCell": ("nvalchemi.dynamics.optimizers", "FIREVariableCell"),
    "SizeAwareSampler": ("nvalchemi.dynamics.sampler", "SizeAwareSampler"),
    "DataSink": ("nvalchemi.dynamics.sinks", "DataSink"),
    "GPUBuffer": ("nvalchemi.dynamics.sinks", "GPUBuffer"),
    "HostMemory": ("nvalchemi.dynamics.sinks", "HostMemory"),
    "ZarrData": ("nvalchemi.dynamics.sinks", "ZarrData"),
    "initialize_velocities": (
        "nvalchemi.dynamics._ops.thermostat_utils",
        "initialize_velocities",
    ),
}
_MODULE_EXPORTS = {
    "hooks": "nvalchemi.dynamics.hooks",
    "integrators": "nvalchemi.dynamics.integrators",
    "optimizers": "nvalchemi.dynamics.optimizers",
}

__all__ = [
    "BaseDynamics",
    "ConvergenceHook",
    "DataSink",
    "DemoDynamics",
    "DistributedPipeline",
    "DynamicsStage",
    "FIRE",
    "FIRE2",
    "FIRE2VariableCell",
    "FIREVariableCell",
    "FusedStage",
    "GPUBuffer",
    "Hook",
    "HostMemory",
    "NPH",
    "NPT",
    "NVE",
    "NVTLangevin",
    "NVTNoseHoover",
    "SizeAwareSampler",
    "ZarrData",
    "hooks",
    "initialize_velocities",
    "integrators",
    "optimizers",
]


def __getattr__(name: str) -> object:
    """Resolve public classes and subpackages only when requested."""
    if name in _MODULE_EXPORTS:
        module = import_module(_MODULE_EXPORTS[name])
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module_name, symbol = _EXPORTS[name]
        value = getattr(import_module(module_name), symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
