from __future__ import annotations

from prbench.memory_paths import MemoryPathStudyConfig, _execution_policy, build_tasks
from prbench.models import DatasetSpec, Distribution, DType, QuantizationConfig, QuantizationMode


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        size=1024,
        dtype=DType.float32,
        distribution=Distribution.uniform,
        seed=1,
        low=-1.0,
        high=1.0,
        quantization=QuantizationConfig(mode=QuantizationMode.binary_fraction, bits=8),
    )


def test_memory_path_task_expansion_is_deterministic() -> None:
    cfg = MemoryPathStudyConfig(
        datasets=[_dataset()],
        operations=["sum"],
        modes=["explicit_sync", "async_pipeline"],
        gpu_ids=[0],
        reuse_counts=[1, 4],
        chunk_elements=[256, 512],
        pipeline_streams=[2, 4],
        blocks=2,
        include_device_resident_diagnostic=False,
    )
    first = build_tasks(cfg)
    second = build_tasks(cfg)
    assert first == second
    assert len(first) == 20


def test_device_resident_diagnostic_is_added_once() -> None:
    cfg = MemoryPathStudyConfig(
        datasets=[_dataset()],
        operations=["sum"],
        modes=["explicit_sync"],
        gpu_ids=[0],
        reuse_counts=[1],
        chunk_elements=[256],
        pipeline_streams=[2],
        blocks=1,
        include_device_resident_diagnostic=True,
    )
    modes = [task.mode for task in build_tasks(cfg)]
    assert modes.count("device_resident") == 1
    assert modes.count("explicit_sync") == 1


def test_host_resident_fraction_is_routing_boundary_not_dataset_cap() -> None:
    cfg = MemoryPathStudyConfig(
        datasets=[_dataset()],
        host_resident_ram_fraction=0.75,
    )
    assert cfg.host_resident_ram_fraction == 0.75


def test_dataset_larger_than_ram_routes_streaming_modes_out_of_core() -> None:
    cfg = MemoryPathStudyConfig(
        datasets=[_dataset()],
        operations=["sum"],
        modes=["chunked_sync", "zero_copy"],
        gpu_ids=[0],
        reuse_counts=[1],
        chunk_elements=[256],
        pipeline_streams=[2],
        blocks=1,
        include_device_resident_diagnostic=False,
    )
    tasks = {task.mode: task for task in build_tasks(cfg)}
    policy, reason = _execution_policy(tasks["chunked_sync"], 2_000, 1_000)
    assert policy == "file_stream"
    assert reason is None

    policy, reason = _execution_policy(tasks["zero_copy"], 2_000, 1_000)
    assert policy == "unsupported_out_of_core"
    assert reason is not None
