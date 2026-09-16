from __future__ import annotations

from pathlib import Path

from prbench.catalog import AlgorithmCatalog
from prbench.config import ConfigurationLoader


def test_multi_gpu_and_hybrid_streaming_are_core_algorithms() -> None:
    catalog = AlgorithmCatalog()

    multi = catalog.get("gpu_multi_cub_file_stream_async")
    assert multi.supports_multi_gpu is True
    assert multi.min_gpus == 2
    assert multi.storage_policy == "file_stream"
    assert multi.memory_path == "async_pipeline"

    hybrid = catalog.get("hybrid_file_stream_equal")
    assert hybrid.uses_cpu is True
    assert hybrid.uses_gpu is True
    assert hybrid.supports_multi_gpu is True
    assert hybrid.storage_policy == "file_stream"


def test_unified_apl13_smoke_uses_normal_root_config_schema() -> None:
    root = Path(__file__).resolve().parent.parent
    config = ConfigurationLoader(AlgorithmCatalog()).load(
        root / "configs/experimental/apl13_integrated_smoke_v4.yaml"
    )
    assert config.measurement.blocks == 1
    assert config.measurement.max_dataset_storage_fraction_of_free == 0.90
    assert any(
        request.id == "gpu_multi_cub_file_stream_async"
        for group in config.experiments
        for request in group.algorithms
    )
    assert any(
        request.id == "gpu_cub_managed_prefetch"
        for group in config.experiments
        for request in group.algorithms
    )


def test_integrated_smoke_requests_one_and_two_gpu_same_machine_baselines() -> None:
    root = Path(__file__).resolve().parent.parent
    config = ConfigurationLoader(AlgorithmCatalog()).load(
        root / "configs/experimental/apl13_integrated_smoke_v4.yaml"
    )
    group = next(group for group in config.experiments if group.id == "v4_host_topologies")
    assert 0 in group.hardware.gpu_sets
    assert 1 in group.hardware.gpu_sets
    assert [0, 1] in group.hardware.gpu_sets
