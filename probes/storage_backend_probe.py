"""TorchStorageBackend semantic smoke without importing the nvalchemi package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "nvalchemi_storage_backend_probe",
    ROOT / "packages/framework/nvalchemi/data/storage_backend.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TorchStorageBackend = module.TorchStorageBackend

backend = TorchStorageBackend()

# Uniform rows: preserve source order, use the first empty destination slots,
# and compact the rows that were not copied.
source = torch.arange(12, dtype=torch.float64).reshape(4, 3)
source_mask = torch.tensor([True, False, True, True])
dest = torch.full((3, 3), -1.0, dtype=torch.float64)
dest_mask = torch.tensor([True, False, False])
copied = torch.zeros(4, dtype=torch.bool)
fit = torch.zeros_like(copied)
backend.compute_put_fit_mask_per_system(source_mask, dest_mask, fit)
assert fit.tolist() == [True, False, True, False]
backend.put_masked_per_system(source, source_mask, dest, dest_mask, copied)
assert copied.tolist() == [True, False, True, False]
assert dest_mask.tolist() == [True, True, True]
assert torch.equal(dest[1:], source[[0, 2]])
backend.defrag_per_system(source, copied)
assert torch.equal(source[:2], torch.tensor([[3., 4., 5.], [9., 10., 11.]], dtype=torch.float64))
assert torch.equal(source[2:], torch.zeros((2, 3), dtype=torch.float64))

# Segmented put: masked segments [2, 1] are appended and their pointers are
# written after the existing destination segment prefix.
seg_source = torch.arange(12, dtype=torch.float64).reshape(6, 2)
seg_ptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
seg_mask = torch.tensor([True, False, True])
seg_dest = torch.zeros((10, 2), dtype=torch.float64)
seg_dest_ptr = torch.zeros(5, dtype=torch.int32)
seg_copied = torch.zeros(3, dtype=torch.bool)
new_num = backend.put_masked_segmented(
    seg_source, seg_ptr, seg_mask, seg_dest, seg_dest_ptr, 0, seg_copied
)
assert new_num is not None and new_num.tolist() == [2]
assert seg_copied.tolist() == [True, False, True]
assert seg_dest_ptr[:3].tolist() == [0, 2, 3]
assert torch.equal(seg_dest[:2], seg_source[:2])
assert torch.equal(seg_dest[2:3], seg_source[5:6])

# Segmented defrag and index expansion preserve kept segment order and pointer
# semantics, including an empty segment.
defrag_source = torch.arange(6, dtype=torch.float64).reshape(6, 1)
defrag_ptr = torch.tensor([0, 0, 3, 4, 6], dtype=torch.int32)
defrag_copied = torch.tensor([True, False, True, False])
kept = backend.defrag_segmented(defrag_source, defrag_ptr, defrag_copied)
assert kept.tolist() == [2]
assert defrag_ptr.tolist() == [0, 3, 5, 5, 5]
assert torch.equal(defrag_source[:5, 0], torch.tensor([0., 1., 2., 4., 5.]))
assert torch.equal(defrag_source[5:], torch.zeros((1, 1), dtype=torch.float64))

expanded = backend.expand_segments(
    torch.tensor([1, 3], dtype=torch.int64),
    torch.tensor([0, 0, 3, 4, 6], dtype=torch.int32),
    index_dtype=torch.int64,
)
assert expanded.tolist() == [0, 1, 2, 4, 5]

print({"status": "passed", "backend": backend.name, "dtype": "torch.float64"})
