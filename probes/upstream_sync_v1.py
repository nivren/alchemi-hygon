# SPDX-License-Identifier: Apache-2.0
"""Exercise unmodified candidate modules through explicit isolated file imports.

This probe validates GroupLayout kernels and Hook data/protocol objects only.
ProbeBatch is a minimal input fixture, not the candidate public Batch API.
The full candidate package import, grouped workflows and Warp remain separate.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from dataclasses import FrozenInstanceError
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch

FRAMEWORK_SHA = "f59e28dc6701583a13bb294e536076c2a7379e98"
SOURCE_HASHES = {
    "nvalchemi/data/group_layout.py": "bb682003ceb8bf82c2201f5e5864960a9baeca8777e2512a54c6794848df9b8a",
    "nvalchemi/hooks/_context.py": "c95b4171d89e977bf6a452f5a72c6867b38677d70c322840112b01c1bf92e35d",
    "nvalchemi/hooks/_protocol.py": "8a747bf878af4f50460599a3e7cf5ab030eb61004bf2aafface07fae75643ba6",
}


def load_module(root, path, name):
    source = root / path
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_HASHES[path]:
        raise ValueError(f"Candidate source differs from frozen SHA: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ProbeBatch:
    def __init__(self, groups, counts, device, occupied=None):
        self.device = device
        self.group_idx = torch.tensor(groups, dtype=torch.long, device=device)
        self.num_graphs = len(counts)
        self.batch_idx = torch.repeat_interleave(
            torch.arange(len(counts), device=device),
            torch.tensor(counts, dtype=torch.long, device=device)
        )
        self._system_group = SimpleNamespace(_num_kept=occupied)

    def __contains__(self, key):
        return key == "group_idx"


def expect_error(error, call):
    try:
        call()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested HCU device is unavailable")
    if any(n == "nvalchemi" or n.startswith("nvalchemi.") or
           n == "nvalchemiops" or n.startswith("nvalchemiops.") for n in sys.modules):
        raise RuntimeError("Product/candidate namespace already imported")
    root = args.framework_root.resolve()
    group = load_module(root, "nvalchemi/data/group_layout.py", "_sync_v1_group")
    context = load_module(root, "nvalchemi/hooks/_context.py", "_sync_v1_context")
    protocol = load_module(root, "nvalchemi/hooks/_protocol.py", "_sync_v1_protocol")
    passed = []

    def equal(value, values):
        assert value.device == device
        torch.testing.assert_close(value, torch.tensor(values, dtype=value.dtype, device=device))

    batch = ProbeBatch([0, 0, 1, 1, 1], [2, 1, 1, 2, 1], device)
    layout = group.GroupLayout.from_batch(batch)
    equal(layout.group_idx, [0, 0, 1, 1, 1])
    equal(layout.graph_rank, [0, 1, 0, 1, 2])
    equal(layout.node_to_group, [0, 0, 0, 1, 1, 1, 1])
    equal(layout.group_ptr, [0, 2, 5])
    equal(layout.num_graphs_per_group, [2, 3])
    assert layout.num_groups == 2 and layout.device == device
    passed.append("group mappings and device")

    mask = torch.tensor([True, True, True, False, True], device=device)
    equal(layout.reduce_all(mask), [True, False])
    equal(layout.reduce_any(mask), [True, True])
    passed.append("group all/any reductions")

    for dtype in (torch.float32, torch.float64):
        values = torch.tensor([[1.5, 2], [3, 4.5]], dtype=dtype, device=device, requires_grad=True)
        actual = layout.broadcast(values)
        equal(actual, [[1.5, 2], [1.5, 2], [3, 4.5], [3, 4.5], [3, 4.5]])
        actual.sum().backward()
        equal(values.grad, [[2, 2], [3, 3]])
        passed.append(f"broadcast and first gradient {dtype}")

    raw = torch.tensor([4, 4, 1, 1, 1], dtype=torch.int32, device=device)
    equal(group._normalize_and_validate_group_idx(raw, num_graphs=5, device=device), [0, 0, 1, 1, 1])
    repeated = torch.tensor([4, 4, 1, 1, 4], device=device)
    expect_error(ValueError, lambda: group._normalize_and_validate_group_idx(repeated, num_graphs=5, device=device))
    passed.append("label normalization and discontiguous rejection")

    for labels in ([1, 1, 2, 2, 2], [0, 0, 2, 2, 2], [0, 1, 0, 1, 1]):
        invalid = ProbeBatch(labels, [2, 1, 1, 2, 1], device)
        expect_error(ValueError, lambda: group.GroupLayout.from_batch(invalid))
    invalid.group_idx = raw.float()
    expect_error(TypeError, lambda: group.GroupLayout.from_batch(invalid))
    passed.append("invalid group numbering and dtype")

    empty = group.GroupLayout.from_batch(ProbeBatch([], [], device))
    assert empty.num_groups == 0
    equal(empty.group_ptr, [0])
    assert empty.reduce_any(torch.empty(0, dtype=torch.bool, device=device)).numel() == 0
    assert empty.broadcast(torch.empty(0, 2, device=device)).shape == (0, 2)
    passed.append("empty input")

    padded = group.GroupLayout.from_batch(ProbeBatch([0, 0, 1, 1, 1, 9, 9], [2, 1, 1, 2, 1], device, occupied=5))
    equal(padded.group_idx, [0, 0, 1, 1, 1])
    passed.append("occupied row slicing")

    expect_error(TypeError, lambda: layout.reduce_any(torch.ones(5, device=device)))
    expect_error(ValueError, lambda: layout.reduce_all(torch.ones(4, dtype=torch.bool, device=device)))
    expect_error(ValueError, lambda: layout.broadcast(torch.ones(3, device=device)))
    expect_error(FrozenInstanceError, lambda: setattr(layout, "group_idx", raw))
    passed.append("shape/mask checks and frozen dataclass")

    ctx = context.DynamicsContext(batch=batch, active_graph_mask=mask, graduated_mask=~mask, step_count=3)
    assert ctx.batch is batch and ctx.active_graph_mask is mask and ctx.step_count == 3
    assert ctx.graduated_mask is not None and ctx.model is None
    expect_error(TypeError, lambda: context.DynamicsContext(batch))
    passed.append("DynamicsContext keywords, masks and identities")

    class Stage(Enum):
        BEFORE = 0

    class Observer:
        frequency = 1
        stage = Stage.BEFORE

        def __call__(self, value, stage):
            assert value is ctx and stage is self.stage

    observer = Observer()
    assert isinstance(observer, protocol.Hook)
    observer(ctx, Stage.BEFORE)
    passed.append("Hook protocol and direct call")
    if device.type == "cuda":
        torch.cuda.synchronize()
    # These aliases import no actual nvalchemi package and perform no fallback.
    assert not any(n == "nvalchemi" or n == "nvalchemiops" for n in sys.modules)
    result = {
        "status": "passed", "candidate_framework_sha": FRAMEWORK_SHA,
        "source_root": str(root), "source_hashes": SOURCE_HASHES,
        "device": str(device), "dtype": ["float32", "float64"],
        "cases_passed": len(passed), "cases": passed,
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_distribution": importlib.metadata.version("torch"),
        "sys_prefix": sys.prefix, "sys_executable": sys.executable,
        "torch_hip": torch.version.hip,
        "jaxtyping": importlib.metadata.version("jaxtyping"),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "architecture": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        if device.type == "cuda" else None,
        "scope": "Unmodified isolated modules; ProbeBatch fixture; not public Batch, engine dispatch, package dependencies or Warp validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
