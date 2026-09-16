from __future__ import annotations

from prbench.models import DatasetSpec, DType, Distribution, QuantizationConfig, QuantizationMode
from prbench.topology_streams import TopologyStreamStudyConfig, build_tasks


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


def test_topology_stream_task_expansion_is_deterministic() -> None:
    cfg = TopologyStreamStudyConfig(
        datasets=[_dataset()],
        operations=["sum"],
        modes=["pinned_direct", "multi_gpu_async", "hybrid_cpu_gpu"],
        storage_policies=["host_resident", "file_stream"],
        gpu_sets=[[0], [0, 1]],
        chunk_elements=[256],
        pipeline_streams=[2],
        cpu_threads=[8],
        cpu_fractions=[0.25, 0.5],
        blocks=2,
    )
    first = build_tasks(cfg)
    second = build_tasks(cfg)
    assert first == second
    # per block: pinned_direct=1, multi_gpu_async=4, hybrid_cpu_gpu=8
    assert len(first) == 26


def test_pinned_direct_is_host_resident_and_single_gpu_only() -> None:
    cfg = TopologyStreamStudyConfig(
        datasets=[_dataset()],
        operations=["sum"],
        modes=["pinned_direct"],
        storage_policies=["host_resident", "file_stream"],
        gpu_sets=[[0], [0, 1]],
        chunk_elements=[256],
        pipeline_streams=[2],
        cpu_threads=[1],
        cpu_fractions=[0.25],
        blocks=1,
    )
    tasks = build_tasks(cfg)
    assert len(tasks) == 1
    assert tasks[0].storage_policy == "host_resident"
    assert tasks[0].gpu_ids == (0,)
