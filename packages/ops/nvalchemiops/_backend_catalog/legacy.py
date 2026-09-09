# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata for the framework-owned upstream legacy implementation."""

from nvalchemiops._backend_types import LEGACY_IMPLEMENTATION_ID, Implementation

IMPLEMENTATIONS: tuple[Implementation, ...] = (
    Implementation(
        implementation_id=LEGACY_IMPLEMENTATION_ID,
        operation="*",
        family="warp",
        executor=None,
        entrypoints=(),
        executor_owner="framework",
        evidence="locked upstream default semantics",
    ),
)
