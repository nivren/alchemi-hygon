from __future__ import annotations

import pytest
import torch

from nvalchemiops._hip_batch_neighbor_runtime import (
    NATIVE_HIP_BATCH_NEIGHBOR_ABI,
    NativeHipNeighborPipelineRequest,
    run_native_hip_batch_neighbor_into,
    validate_native_hip_neighbor_pipeline,
)
from nvalchemiops.backend import BackendUnavailableError


def _request(**overrides: object) -> NativeHipNeighborPipelineRequest:
    values: dict[str, object] = {
        "device_type": "cuda",
        "dtype": "float32",
    }
    values.update(overrides)
    return NativeHipNeighborPipelineRequest(**values)


def test_native_hip_neighbor_runtime_contract_describes_full_fixed_torch_path():
    abi = validate_native_hip_neighbor_pipeline(_request())

    assert abi is NATIVE_HIP_BATCH_NEIGHBOR_ABI
    assert abi.build_stage == "checked_init_then_trusted_reusable_csr"
    assert abi.query_stage == "native_unordered_candidate_query"
    assert abi.topology_stage == "native_composite_key_to_canonical_full_matrix"
    assert abi.geometry_stage == "torch_reference_distance_vector"
    assert abi.geometry_gradient_order == 2
    assert abi.supports_skin is False
    assert abi.native_geometry_registered is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("device_type", "cpu", "HIP-capable Torch device"),
        ("dtype", "float16", "float32 and float64"),
        ("batch", False, "Batch input"),
        ("fixed_cell", False, "variable-cell"),
        ("full_list", False, "half-list"),
        ("geometry_backend", "native_hip", "Torch reference geometry"),
        ("native_geometry", True, "forward-only candidate"),
        ("skin", 0.5, "skin/rebuild"),
        ("rebuild", True, "skin/rebuild"),
        ("target_indices", True, "target-index"),
        ("pair_outputs", True, "pair-function"),
        ("distributed", True, "distributed/domain-parallel"),
    ],
)
def test_native_hip_neighbor_runtime_contract_rejects_unsupported_requests(
    field: str, value: object, reason: str
):
    with pytest.raises(BackendUnavailableError, match=reason):
        validate_native_hip_neighbor_pipeline(_request(**{field: value}))


def test_native_hip_neighbor_runtime_contract_allows_torch_geometry_gradients():
    for gradient_order in (0, 1, 2):
        abi = validate_native_hip_neighbor_pipeline(
            _request(gradient_order=gradient_order)
        )
        assert abi.geometry_gradient_order >= gradient_order


def test_native_hip_neighbor_runtime_contract_is_not_registry_registration():
    """The ABI object must not imply that backend='hip' is selectable yet."""
    assert NATIVE_HIP_BATCH_NEIGHBOR_ABI.native_geometry_registered is False


def test_native_hip_neighbor_runtime_wrapper_rejects_cpu_before_mutation():
    class Plan:
        positions = torch.zeros((1, 3), dtype=torch.float32)
        pbc = torch.ones((1, 3), dtype=torch.bool)
        batch_idx = torch.zeros((1,), dtype=torch.int32)
        cells_per_dimension = torch.ones((1, 3), dtype=torch.int32)
        cell_offsets = torch.zeros((1,), dtype=torch.int32)
        atom_periodic_shifts = torch.zeros((1, 3), dtype=torch.int32)
        atom_to_cell_mapping = torch.zeros((1, 3), dtype=torch.int32)
        cell_counts = torch.zeros((1,), dtype=torch.int32)
        cell_starts = torch.zeros((1,), dtype=torch.int32)
        cell_atom_list = torch.zeros((1,), dtype=torch.int32)
        run_called = False

        def run(self):
            self.run_called = True

    plan = Plan()
    # The admission gate must reject before it inspects or mutates any of the
    # twelve explicit output/workspace arguments.
    buffers = (object(),) * 12

    with pytest.raises(BackendUnavailableError, match="HIP-capable Torch device"):
        run_native_hip_batch_neighbor_into(plan, *buffers)
    assert plan.run_called is False
