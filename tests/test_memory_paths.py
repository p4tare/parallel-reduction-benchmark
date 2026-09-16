from __future__ import annotations

from types import SimpleNamespace

from prbench.capacity import gpu_input_allocation_estimates
from prbench.catalog import AlgorithmCatalog
from prbench.models import DatasetSpec, DType, Distribution, QuantizationConfig, QuantizationMode


def _dataset(size: int = 1024) -> DatasetSpec:
    return DatasetSpec(
        size=size,
        dtype=DType.float32,
        distribution=Distribution.uniform,
        seed=1,
        low=-1.0,
        high=1.0,
        quantization=QuantizationConfig(mode=QuantizationMode.binary_fraction, bits=8),
    )


def _task(algorithm_id: str, params: dict[str, object] | None = None) -> SimpleNamespace:
    algorithm = AlgorithmCatalog().get(algorithm_id)
    return SimpleNamespace(
        algorithm=algorithm,
        algorithm_params=params or {},
        dataset=_dataset(10_000_000),
        gpu_ids=[0],
        task_key="test",
    )


def test_new_memory_paths_live_in_core_algorithm_catalog() -> None:
    catalog = AlgorithmCatalog()
    expected = {
        "gpu_cub_chunked": ("chunked_sync", "host_resident"),
        "gpu_cub_pinned_direct": ("pinned_direct", "host_resident"),
        "gpu_cub_zero_copy": ("zero_copy", "host_resident"),
        "gpu_cub_managed_fault": ("managed_fault", "host_resident"),
        "gpu_cub_managed_prefetch": ("managed_prefetch", "host_resident"),
        "gpu_cub_managed_advised": ("managed_advised", "host_resident"),
        "gpu_cub_hmm": ("hmm_system", "host_resident"),
        "gpu_cub_file_stream": ("chunked_sync", "file_stream"),
        "gpu_cub_file_stream_async": ("async_pipeline", "file_stream"),
    }
    for algorithm_id, (memory_path, storage_policy) in expected.items():
        definition = catalog.get(algorithm_id)
        assert definition.memory_path == memory_path
        assert definition.storage_policy == storage_policy


def test_host_mapped_and_managed_paths_do_not_require_full_input_vram() -> None:
    for algorithm_id in (
        "gpu_cub_zero_copy",
        "gpu_cub_managed_fault",
        "gpu_cub_managed_prefetch",
        "gpu_cub_managed_advised",
        "gpu_cub_hmm",
    ):
        estimate = gpu_input_allocation_estimates(_task(algorithm_id))[0]
        assert estimate["input_bytes"] == 0
        assert estimate["kind"] == "managed_or_host_mapped"


def test_file_stream_vram_estimate_is_bounded_by_pipeline_not_dataset() -> None:
    task = _task(
        "gpu_cub_file_stream_async",
        {"pipeline_streams": 2, "pipeline_chunk_elements": 1_048_576},
    )
    estimate = gpu_input_allocation_estimates(task)[0]
    assert estimate["kind"] == "exact"
    assert estimate["elements"] == 2 * 1_048_576
    assert estimate["input_bytes"] == 2 * 1_048_576 * 4


def test_out_of_core_cpu_baseline_is_part_of_core_catalog() -> None:
    definition = AlgorithmCatalog().get("cpu_omp_simd_file_stream")
    assert definition.uses_cpu is True
    assert definition.uses_gpu is False
    assert definition.storage_policy == "file_stream"
