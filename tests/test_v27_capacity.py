from pathlib import Path

from prbench.capacity import (
    cache_rotation_replicas,
    cache_rotated_resident_bytes,
    dataset_size_bytes,
    gpu_capacity_rows,
)
from prbench.catalog import AlgorithmCatalog
from prbench.config import ConfigurationLoader
from prbench.models import DType, DatasetSpec, ExperimentGroup, HardwareConfig, RootConfig, SystemTopologyModel, TopologyCpu, TopologyGpu
from prbench.sweep import SweepPlanner


def topology() -> SystemTopologyModel:
    return SystemTopologyModel(
        hostname="x", os="Linux", kernel="x", machine="x86_64",
        logical_cpus=[
            TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0),
            TopologyCpu(cpu_id=1, socket_id=0, core_id=1, numa_node=0),
        ],
        allowed_cpus=[0, 1], numa_nodes={0: [0, 1]},
        gpus=[
            TopologyGpu(index=0, name="g0", uuid="0", pci_bus_id="0000:01:00.0", memory_bytes=16<<30, memory_free_bytes=15<<30),
            TopologyGpu(index=1, name="g1", uuid="1", pci_bus_id="0000:02:00.0", memory_bytes=16<<30, memory_free_bytes=15<<30),
        ],
        total_ram_bytes=64 << 30, nvml_available=True,
    )


def test_dataset_size_bytes() -> None:
    assert dataset_size_bytes(DatasetSpec(size=100, dtype=DType.float32)) == 400
    assert dataset_size_bytes(DatasetSpec(size=100, dtype=DType.int64)) == 800


def test_equal_multi_gpu_capacity_is_partitioned() -> None:
    cfg = RootConfig(
        measurement={"blocks": 1, "timing_repetitions": 3},
        energy={"enable_cpu": False, "enable_gpu": False},
        experiments=[ExperimentGroup(
            id="m", datasets=[DatasetSpec(size=1_000_000, dtype=DType.int64)],
            algorithms=[{"id": "gpu_multi_cub_equal"}],
            hardware=HardwareConfig(gpu_sets=["all"]),
        )],
    )
    task = SweepPlanner(AlgorithmCatalog(), topology()).plan(cfg)[0]
    rows = gpu_capacity_rows(task, topology(), 0.8)
    assert len(rows) == 2
    assert {r["estimate_kind"] for r in rows} == {"exact"}
    assert {r["estimated_input_bytes"] for r in rows} == {4_000_000}


def test_cuda_visible_devices_identity_check(monkeypatch) -> None:
    from prbench.cli import _cuda_visible_devices_problem
    t = topology()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert _cuda_visible_devices_problem(t) is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert "masks/reorders" in (_cuda_visible_devices_problem(t) or "")


def test_gpu_only_async_capacity_uses_staging_slots() -> None:
    cfg = RootConfig(
        measurement={"blocks": 1, "timing_repetitions": 3},
        energy={"enable_cpu": False, "enable_gpu": False},
        experiments=[ExperimentGroup(
            id="a", datasets=[DatasetSpec(size=16_000_000, dtype=DType.float32)],
            algorithms=[{"id": "gpu_cub_async", "params": {"pipeline_streams": 2, "pipeline_chunks": 8}}],
            hardware=HardwareConfig(gpu_sets=["each"]),
        )],
    )
    task = SweepPlanner(AlgorithmCatalog(), topology()).plan(cfg)[0]
    rows = gpu_capacity_rows(task, topology(), 0.8)
    # 16M / 8 chunks * 2 staging slots * 4 B = 16 MB of device input buffers.
    assert rows[0]["estimated_input_bytes"] == 16_000_000
    assert rows[0]["estimate_kind"] == "exact"


def test_cache_rotation_memory_accounting() -> None:
    assert cache_rotation_replicas(4_000_000, 256_000_000, 1024) == 64
    assert cache_rotated_resident_bytes(4_000_000, 256_000_000, 1024) == 256_000_000
    assert cache_rotation_replicas(400_000_000, 256_000_000, 1024) == 1


def test_gpu_only_async_fixed_chunk_capacity_is_bounded() -> None:
    cfg = RootConfig(
        measurement={"blocks": 1, "timing_repetitions": 3},
        energy={"enable_cpu": False, "enable_gpu": False},
        experiments=[ExperimentGroup(
            id="bounded_async",
            datasets=[DatasetSpec(size=1_000_000_000, dtype=DType.float64)],
            algorithms=[{"id": "gpu_cub_async", "params": {
                "pipeline_streams": 2,
                "pipeline_chunk_elements": 16_777_216,
            }}],
            hardware=HardwareConfig(gpu_sets=[0]),
        )],
    )
    task = SweepPlanner(AlgorithmCatalog(), topology()).plan(cfg)[0]
    rows = gpu_capacity_rows(task, topology(), 0.8)
    assert rows[0]["estimated_input_bytes"] == 268_435_456
    assert rows[0]["estimate_kind"] == "exact"


def test_final_v3_configs_validate_and_plan_on_two_gpus() -> None:
    root = Path(__file__).resolve().parents[1]
    loader = ConfigurationLoader(AlgorithmCatalog())
    planner = SweepPlanner(AlgorithmCatalog(), topology())
    names = [
        "FINAL_VALIDATION_v3.yaml",
        "FINAL_TUNING_v3.yaml",
        "FINAL_RESEARCH_ALGORITHMS_v3.yaml",
        "FINAL_RESEARCH_SCALING_v3.yaml",
    ]
    for name in names:
        cfg = loader.load(root / "configs" / name)
        tasks = planner.plan(cfg)
        assert tasks, name
