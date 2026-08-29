from __future__ import annotations

import json
from pathlib import Path

from prbench.catalog import AlgorithmCatalog
from prbench.cli import _design_warnings
from prbench.config import ConfigurationLoader
from prbench.models import MeasurementConfig
from prbench.results import ResultsStore
from prbench.telemetry import TelemetryCollector


def test_auto_timing_configuration_is_valid() -> None:
    cfg = MeasurementConfig(
        timing_repetitions="auto",
        timing_probe_repetitions=7,
        timing_target_batch_seconds=0.25,
        timing_min_repetitions=20,
        timing_max_repetitions=20000,
    )
    assert cfg.timing_repetitions == "auto"
    assert cfg.timing_probe_repetitions == 7


def test_fixed_timing_remains_backward_compatible() -> None:
    cfg = MeasurementConfig(timing_repetitions=3)
    assert cfg.timing_repetitions == 3


def test_all_options_template_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ConfigurationLoader(AlgorithmCatalog()).load(root / "configs" / "CONFIG_ALL_OPTIONS_TEMPLATE.yaml")
    assert cfg.measurement.timing_repetitions == "auto"
    assert {op.value for op in cfg.experiments[0].operations} == {"sum", "min", "max"}
    assert cfg.telemetry.enabled is True


def test_manifest_is_finalized_with_end_timestamp(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    store.write_manifest({"timestamp_start": "2026-01-01T00:00:00+00:00"})
    store.append_task({"status": "ok"})
    store.finalize_manifest(status="completed")
    data = json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["timestamp_end"]
    assert data["run_duration_s"] >= 0
    assert data["run_status"] == "completed"
    assert data["task_status_counts"] == {"ok": 1}


def test_telemetry_disabled_is_non_intrusive() -> None:
    cfg = type(
        "Cfg",
        (),
        {
            "enabled": False,
            "capture_cpu_frequency": True,
            "capture_cpu_temperature": True,
            "capture_gpu_state": True,
        },
    )()
    row = TelemetryCollector(cfg).snapshot("test", [], [])
    assert row["phase"] == "test"
    assert row["enabled"] is False
    assert row["timestamp"]


def test_preflight_design_warning_catches_operation_confounding(tmp_path: Path) -> None:
    # Use a tiny synthetic config and a minimal homogeneous topology through planner-independent
    # task-like objects; _design_warnings intentionally depends only on task metadata.
    class Dataset:
        size = 1000
        distribution = type("D", (), {"value": "uniform"})()
        dtype = type("T", (), {"value": "float32"})()

    class Algo:
        id = "cpu_omp_simd"

    def task(core_class: str, operation: str):
        return type(
            "Task",
            (),
            {
                "group_id": "g",
                "algorithm": Algo(),
                "dataset": Dataset(),
                "gpu_ids": [],
                "algorithm_params": {},
                "cpu_thread_policy": "all_threads",
                "cpu_core_class": core_class,
                "gpu_control_mode": "dedicated",
                "operation": type("O", (), {"value": operation})(),
            },
        )()

    warnings = _design_warnings([task("performance", "sum"), task("efficiency", "max")])
    assert any("confounding" in item and "CPU core classes" in item for item in warnings)


def test_thesis_preflight_and_cache_rotation_controls_validate() -> None:
    cfg = MeasurementConfig(
        timing_repetitions="auto",
        cache_rotation_target_bytes=256 * 1024 * 1024,
        cache_rotation_max_replicas=64,
        strict_preflight=True,
        max_preflight_cpu_load_percent=5.0,
        allow_gpu_graphics_processes=False,
    )
    assert cfg.cache_rotation_target_bytes == 256 * 1024 * 1024
    assert cfg.cache_rotation_max_replicas == 64
    assert cfg.strict_preflight is True
