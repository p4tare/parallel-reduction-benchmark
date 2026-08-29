from prbench.catalog import AlgorithmCatalog
from prbench.models import (
    DType,
    DatasetSpec,
    ExperimentGroup,
    HardwareConfig,
    RootConfig,
    SystemTopologyModel,
    TopologyCpu,
)
from prbench.sweep import SweepPlanner


def test_cpu_only_does_not_require_gpu() -> None:
    topology = SystemTopologyModel(
        hostname="x",
        os="Linux",
        kernel="x",
        machine="x86_64",
        logical_cpus=[TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0)],
        allowed_cpus=[0],
        numa_nodes={0: [0]},
        gpus=[],
        total_ram_bytes=1 << 30,
        nvml_available=False,
    )
    config = RootConfig(
        experiments=[
            ExperimentGroup(
                id="x",
                datasets=[DatasetSpec(size=10, dtype=DType.int32)],
                algorithms=[{"id": "cpu_seq"}],
                hardware=HardwareConfig(gpu_sets=[]),
            )
        ]
    )
    tasks = SweepPlanner(AlgorithmCatalog(), topology).plan(config)
    assert tasks
    assert tasks[0].gpu_ids == []


def test_core_class_and_thread_policy_selection() -> None:
    from prbench.topology import resolve_cpu_pool

    topology = SystemTopologyModel(
        hostname="x", os="Linux", kernel="x", machine="x86_64",
        logical_cpus=[
            TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0, core_class="performance"),
            TopologyCpu(cpu_id=1, socket_id=0, core_id=0, numa_node=0, core_class="performance"),
            TopologyCpu(cpu_id=2, socket_id=0, core_id=1, numa_node=0, core_class="efficiency"),
            TopologyCpu(cpu_id=3, socket_id=0, core_id=2, numa_node=0, core_class="efficiency"),
        ],
        allowed_cpus=[0, 1, 2, 3], numa_nodes={0: [0, 1, 2, 3]}, gpus=[],
        total_ram_bytes=1 << 30, nvml_available=False,
    )
    assert resolve_cpu_pool(topology, "performance", "all_threads", None) == [0, 1]
    assert resolve_cpu_pool(topology, "performance", "one_thread_per_core", None) == [0]
    assert resolve_cpu_pool(topology, "efficiency", "all_threads", None) == [2, 3]
    assert resolve_cpu_pool(topology, "all", "all_threads", None) == [0, 1, 2, 3]


def test_gpu_control_threads_use_distinct_physical_cores() -> None:
    from prbench.models import TopologyGpu

    topology = SystemTopologyModel(
        hostname="x", os="Linux", kernel="x", machine="x86_64",
        logical_cpus=[
            TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0),
            TopologyCpu(cpu_id=1, socket_id=0, core_id=0, numa_node=0),
            TopologyCpu(cpu_id=2, socket_id=0, core_id=1, numa_node=0),
            TopologyCpu(cpu_id=3, socket_id=0, core_id=1, numa_node=0),
            TopologyCpu(cpu_id=4, socket_id=0, core_id=2, numa_node=0),
        ],
        allowed_cpus=[0, 1, 2, 3, 4], numa_nodes={0: [0, 1, 2, 3, 4]},
        gpus=[
            TopologyGpu(index=0, name="g0", uuid="0", pci_bus_id="0000:01:00.0", memory_bytes=1, local_cpus=[0,1,2,3,4]),
            TopologyGpu(index=1, name="g1", uuid="1", pci_bus_id="0000:02:00.0", memory_bytes=1, local_cpus=[0,1,2,3,4]),
        ],
        total_ram_bytes=1 << 30, nvml_available=True,
    )
    planner = SweepPlanner(AlgorithmCatalog(), topology)
    remaining, workers, bindings = planner._assign_gpu_control_cpus([0, 2, 4], [0, 1], "shared")
    assert workers == [0, 2]
    assert remaining == [0, 2, 4]
    assert [(b["socket_id"], b["core_id"]) for b in bindings] == [(0, 0), (0, 1)]

    remaining, workers, _ = planner._assign_gpu_control_cpus([0, 1, 2, 3, 4], [0, 1], "dedicated")
    assert workers == [0, 2]
    assert remaining == [4]


def test_gpu_control_threads_prefer_each_gpus_local_numa_core() -> None:
    from prbench.models import TopologyGpu

    topology = SystemTopologyModel(
        hostname="dual", os="Linux", kernel="x", machine="x86_64",
        logical_cpus=[
            TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0),
            TopologyCpu(cpu_id=1, socket_id=0, core_id=1, numa_node=0),
            TopologyCpu(cpu_id=2, socket_id=1, core_id=0, numa_node=1),
            TopologyCpu(cpu_id=3, socket_id=1, core_id=1, numa_node=1),
        ],
        allowed_cpus=[0, 1, 2, 3], numa_nodes={0: [0, 1], 1: [2, 3]},
        gpus=[
            TopologyGpu(
                index=0, name="g0", uuid="0", pci_bus_id="0000:01:00.0",
                memory_bytes=1, numa_node=0, local_cpus=[0, 1]
            ),
            TopologyGpu(
                index=1, name="g1", uuid="1", pci_bus_id="0000:81:00.0",
                memory_bytes=1, numa_node=1, local_cpus=[2, 3]
            ),
        ],
        total_ram_bytes=1 << 30, nvml_available=True,
    )
    planner = SweepPlanner(AlgorithmCatalog(), topology)
    remaining, workers, bindings = planner._assign_gpu_control_cpus(
        [0, 1, 2, 3], [0, 1], "dedicated"
    )
    assert workers == [0, 2]
    assert [b["numa_node"] for b in bindings] == [0, 1]
    assert remaining == [1, 3]


def test_operations_are_first_class_sweep_dimension() -> None:
    from prbench.models import ReductionOperation

    topology = SystemTopologyModel(
        hostname="x", os="Linux", kernel="x", machine="x86_64",
        logical_cpus=[TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0)],
        allowed_cpus=[0], numa_nodes={0: [0]}, gpus=[], total_ram_bytes=1 << 30,
        nvml_available=False,
    )
    config = RootConfig(
        measurement={"blocks": 1, "timing_repetitions": 3},
        energy={"enable_cpu": False, "enable_gpu": False},
        experiments=[
            ExperimentGroup(
                id="ops",
                datasets=[DatasetSpec(size=10, dtype=DType.int32)],
                algorithms=[{"id": "cpu_seq"}],
                operations=[ReductionOperation.sum, ReductionOperation.min, ReductionOperation.max],
                hardware=HardwareConfig(gpu_sets=[]),
            )
        ],
    )
    tasks = SweepPlanner(AlgorithmCatalog(), topology).plan(config)
    assert {task.operation for task in tasks} == {
        ReductionOperation.sum, ReductionOperation.min, ReductionOperation.max
    }
    assert len(tasks) == 3


def test_parameter_sweep_filters_only_invalid_pipeline_cross_product_pairs() -> None:
    combos = SweepPlanner._expand_params(
        {
            "pipeline_streams": [1, 2, 4, 8],
            "pipeline_chunks": [4, 8, 16, 32, 64],
        }
    )
    assert len(combos) == 19
    assert {
        (int(item["pipeline_streams"]), int(item["pipeline_chunks"])) for item in combos
    } == {
        (streams, chunks)
        for streams in (1, 2, 4, 8)
        for chunks in (4, 8, 16, 32, 64)
        if chunks >= streams
    }


def test_explicit_invalid_pipeline_configuration_still_fails_fast() -> None:
    import pytest

    with pytest.raises(ValueError, match="pipeline_chunks cannot be smaller"):
        SweepPlanner._expand_params({"pipeline_streams": 8, "pipeline_chunks": 4})


def test_parameter_sweep_rejects_when_all_combinations_are_invalid() -> None:
    import pytest

    with pytest.raises(ValueError, match="no valid combinations"):
        SweepPlanner._expand_params(
            {"pipeline_streams": [8, 16], "pipeline_chunks": [1, 2, 4]}
        )


def test_multi_gpu_pairs_selector_expands_all_pairs_and_deduplicates_all() -> None:
    from prbench.models import TopologyGpu

    topology = SystemTopologyModel(
        hostname="tri", os="Linux", kernel="x", machine="x86_64",
        logical_cpus=[
            TopologyCpu(cpu_id=0, socket_id=0, core_id=0, numa_node=0),
            TopologyCpu(cpu_id=1, socket_id=0, core_id=1, numa_node=0),
            TopologyCpu(cpu_id=2, socket_id=0, core_id=2, numa_node=0),
        ],
        allowed_cpus=[0, 1, 2], numa_nodes={0: [0, 1, 2]},
        gpus=[
            TopologyGpu(index=i, name=f"g{i}", uuid=str(i), pci_bus_id=f"0000:0{i+1}:00.0", memory_bytes=8<<30)
            for i in range(3)
        ],
        total_ram_bytes=64 << 30, nvml_available=True,
    )
    config = RootConfig(
        measurement={"blocks": 1, "timing_repetitions": 3},
        energy={"enable_cpu": False, "enable_gpu": False},
        experiments=[
            ExperimentGroup(
                id="pairs",
                datasets=[DatasetSpec(size=1024, dtype=DType.float32)],
                algorithms=[{"id": "gpu_multi_cub_equal"}],
                hardware=HardwareConfig(gpu_sets=["pairs", "all"]),
            )
        ],
    )
    tasks = SweepPlanner(AlgorithmCatalog(), topology).plan(config)
    assert {tuple(t.gpu_ids) for t in tasks} == {(0, 1), (0, 2), (1, 2), (0, 1, 2)}
