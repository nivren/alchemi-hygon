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
"""MD integrators with lazy imports across optional backend boundaries."""

from importlib import import_module

_EXPORTS = {
    "NPH": ("nvalchemi.dynamics.integrators.nph", "NPH"),
    "NPT": ("nvalchemi.dynamics.integrators.npt", "NPT"),
    "NVE": ("nvalchemi.dynamics.integrators.nve", "NVE"),
    "NVTLangevin": ("nvalchemi.dynamics.integrators.nvt_langevin", "NVTLangevin"),
    "NVTNoseHoover": (
        "nvalchemi.dynamics.integrators.nvt_nose_hoover",
        "NVTNoseHoover",
    ),
}

__all__ = [
    "NPH",
    "NPT",
    "NVE",
    "NVTLangevin",
    "NVTNoseHoover",
]


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = _EXPORTS[name]
    value = getattr(import_module(module_name), symbol)
    globals()[name] = value
    return value
