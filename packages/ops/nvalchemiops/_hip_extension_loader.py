# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lazy loader for the optional native HIP extensions.

Each HIP operation keeps its own extension name, cache directory and source
set, but the JIT build mechanics are identical.  Keeping that mechanics here
prevents the individual operation wrappers from drifting in compiler flags or
cache handling.  This helper only loads code on an explicit HIP call; it does
not provide a CPU fallback.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import ModuleType

from torch.utils.cpp_extension import get_default_build_root, load


def load_hip_extension(
    *,
    source_root: Path,
    extension_name: str,
    source_names: tuple[str, str],
    build_env_var: str,
    verbose_env_var: str,
) -> ModuleType:
    """Build and load one native HIP extension into an explicit cache.

    The source files are copied into the cache before compilation so the JIT
    build never depends on a mutable working-tree path after it starts.  The
    caller owns module-level caching and locking because each wrapper exposes
    a different extension and may also support an AOT load path.
    """

    default_build = Path(get_default_build_root()) / extension_name
    build_root = Path(os.environ.get(build_env_var, str(default_build))).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    compile_root = build_root / "sources"
    compile_root.mkdir(parents=True, exist_ok=True)

    compile_sources: list[str] = []
    for filename in source_names:
        source = source_root / filename
        destination = compile_root / filename
        shutil.copy2(source, destination)
        compile_sources.append(str(destination))

    return load(
        name=extension_name,
        sources=compile_sources,
        build_directory=str(build_root),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=os.environ.get(verbose_env_var) == "1",
    )


__all__ = ["load_hip_extension"]
