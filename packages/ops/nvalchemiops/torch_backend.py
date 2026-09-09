# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint for the backend-neutral operation dispatcher."""

from nvalchemiops.backend import resolve_backend
from nvalchemiops.dispatch import dispatch_lj_energy_forces, dispatch_neighbor_list

__all__ = ["dispatch_lj_energy_forces", "dispatch_neighbor_list"]
