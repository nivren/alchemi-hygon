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

"""Lightweight dynamics-stage definitions shared by model and hook setup.

This module intentionally has no imports from the dynamics package.  Model
configuration code can therefore create a hook in a Warp-free reference
environment without importing the eager dynamics package initializer.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["DynamicsStage"]


class DynamicsStage(Enum):
    """
    Enumeration of stages in the dynamics step where hooks can fire.

    Each stage corresponds to a specific point in the simulation step,
    allowing hooks to be triggered before or after key operations.

    Attributes
    ----------
    BEFORE_STEP : int
        Fired at the very beginning of a step, before any operations.
    BEFORE_PRE_UPDATE : int
        Fired before the pre_update (first half of integrator) is called.
    AFTER_PRE_UPDATE : int
        Fired after the pre_update completes.
    BEFORE_COMPUTE : int
        Fired before the model forward pass (force/energy computation).
    AFTER_COMPUTE : int
        Fired after the model forward pass completes.
    BEFORE_POST_UPDATE : int
        Fired before the post_update (second half of integrator) is called.
    AFTER_POST_UPDATE : int
        Fired after the post_update completes.
    AFTER_STEP : int
        Fired at the very end of a step, after all operations.
    ON_CONVERGE : int
        Fired when a convergence criterion is met (e.g., for optimizers).
    """

    BEFORE_STEP = 0
    BEFORE_PRE_UPDATE = 1
    AFTER_PRE_UPDATE = 2
    BEFORE_COMPUTE = 3
    AFTER_COMPUTE = 4
    BEFORE_POST_UPDATE = 5
    AFTER_POST_UPDATE = 6
    AFTER_STEP = 7
    ON_CONVERGE = 8
