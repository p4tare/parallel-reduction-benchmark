from __future__ import annotations

from prbench.memory_paths import MemoryPathStudyConfig, build_tasks
from prbench.models import DatasetSpec, DType, Distribution, QuantizationConfig, QuantizationMode


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
    # explicit: 2 reuse * 1 chunk/stream representative = 2 tasks/block
    # async: 2 reuse * 2 chunks * 2 stream counts = 8 tasks/block
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


def test_large_dataset_fraction_guard_is_configurable() -> None:
    cfg = MemoryPathStudyConfig(
        datasets=[_dataset()],
        max_dataset_ram_fraction=0.75,
    )
    assert cfg.max_dataset_ram_fraction == 0.75
