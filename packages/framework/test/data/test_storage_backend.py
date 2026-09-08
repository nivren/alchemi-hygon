# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for the Torch level-storage backend boundary."""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nvalchemi.data.level_storage import SegmentedLevelStorage, UniformLevelStorage
from nvalchemi.data.storage_backend import TorchStorageBackend


def test_public_data_import_does_not_load_warp() -> None:
    """The Torch data path must remain importable without NVIDIA Warp."""
    framework = Path(__file__).parents[2]
    ops = framework.parent / "ops"
    code = """
import sys
from nvalchemi.data import AtomicData, Batch
assert 'warp' not in sys.modules
print(AtomicData.__name__, Batch.__name__)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{framework}:{ops}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=framework.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "AtomicData Batch"


def test_level_storage_uses_torch_as_the_documented_default() -> None:
    """The data model has a stable HCU-safe default independent of Warp."""
    storage = UniformLevelStorage(data={"x": torch.zeros(1, 1)}, validate=False)
    assert isinstance(storage._backend, TorchStorageBackend)
    assert storage._backend.name == "torch"


@pytest.mark.xfail(
    importlib.util.find_spec("warp") is None,
    strict=True,
    reason="WARP-EQUIV-001: run the explicit storage equivalence path on NVIDIA/Warp",
)
def test_explicit_warp_storage_matches_torch_put_contract() -> None:
    """Warp remains an explicit comparison backend, never an implicit default."""
    from nvalchemi.data.warp_storage_backend import WarpStorageBackend

    source_data = {
        "x": torch.tensor([[1.0], [2.0]]),
        "y": torch.tensor([[10.0], [20.0]]),
    }
    outputs = []
    for backend in (TorchStorageBackend(), WarpStorageBackend()):
        source = UniformLevelStorage(data=source_data, validate=False, backend=backend)
        dest = UniformLevelStorage(
            data={"x": torch.zeros(3, 1), "y": torch.zeros(3, 1)},
            validate=False,
            backend=backend,
        )
        dest.put(source, torch.tensor([True, True]))
        outputs.append((dest["x"].clone(), dest["y"].clone(), source._copied_mask.clone()))
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][1], outputs[1][1])
    assert torch.equal(outputs[0][2], outputs[1][2])


def test_uniform_multi_attribute_put_reuses_slots_and_mask() -> None:
    """Every common attribute must use the same selected destination rows."""
    src = UniformLevelStorage(
        data={
            "x": torch.tensor([[1.0], [2.0]]),
            "y": torch.tensor([[10.0], [20.0]]),
        },
        device="cpu",
        validate=False,
    )
    dest = UniformLevelStorage(
        data={"x": torch.zeros(3, 1), "y": torch.zeros(3, 1)},
        device="cpu",
        validate=False,
    )

    dest.put(src, torch.tensor([True, True]))

    torch.testing.assert_close(dest["x"], torch.tensor([[1.0], [2.0], [0.0]]))
    torch.testing.assert_close(dest["y"], torch.tensor([[10.0], [20.0], [0.0]]))
    assert getattr(src, "_copied_mask").tolist() == [True, True]


def test_segmented_multi_attribute_put_reuses_offsets_and_mask() -> None:
    """Every common segmented attribute must share boundaries and offsets."""
    src = SegmentedLevelStorage(
        data={
            "x": torch.tensor([[1.0], [2.0], [3.0]]),
            "y": torch.tensor([[10.0], [20.0], [30.0]]),
        },
        segment_lengths=[1, 2],
        device="cpu",
        validate=False,
    )
    dest = SegmentedLevelStorage(
        data={"x": torch.zeros(8, 1), "y": torch.zeros(8, 1)},
        segment_lengths=[1],
        batch_ptr_capacity=5,
        device="cpu",
        validate=False,
    )

    dest.put(src, torch.tensor([True, True]))

    assert dest.segment_lengths.tolist() == [1, 1, 2]
    torch.testing.assert_close(dest["x"][:4], torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    torch.testing.assert_close(dest["y"][:4], torch.tensor([[0.0], [10.0], [20.0], [30.0]]))
    assert getattr(src, "_copied_mask").tolist() == [True, True]
