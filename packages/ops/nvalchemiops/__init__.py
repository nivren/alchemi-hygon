# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

__version__ = "0.4.1"

from typing import Any


def initialize_warp() -> Any:
    """Load and initialize Warp for an explicitly selected Warp backend.

    Importing :mod:`nvalchemiops` itself is backend-neutral.  NVIDIA-facing
    subpackages call this function at their boundary, while Torch reference
    code can be imported without installing or initializing Warp.
    """
    import warp as wp

    if not getattr(wp, "_nvalchemiops_initialized", False):
        wp.config.quiet = True
        try:
            wp.init()
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to initialize warp, likely due to missing drivers and/or devices."
                " Make sure you have the correct CUDA version, and that GPUs are available."
            ) from exc
        wp._nvalchemiops_initialized = True
    return wp


__all__ = ["__version__", "initialize_warp"]
