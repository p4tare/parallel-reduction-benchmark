from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from .capacity import cache_rotated_resident_bytes, dataset_size_bytes, gpu_capacity_rows
from .datasets import DatasetArtifact, DatasetFactory
from .energy import CompositeEnergyMeter
from .models import RootConfig, SystemTopologyModel
from .protocol import ProtocolError, read_event, send_command
from .results import ResultsStore
from .sweep import TaskSpec
from .telemetry import TelemetryCollector, utc_now_iso
from .validation import ResultValidator

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


class ThermalGuard:
    def __init__(self, topology: SystemTopologyModel, config: RootConfig) -> None:
        self.topology = topology
        self.max_gpu = config.measurement.thermal_safety_gpu_c
        self.max_cpu = config.measurement.thermal_safety_cpu_c
        self.timeout = config.measurement.thermal_wait_timeout_s

    def wait_until_safe(self, gpu_ids: list[int]) -> None:
        if self.timeout <= 0:
            return
        deadline = time.monotonic() + self.timeout
        while True:
            cpu_t = self._max_cpu_temp()
            gpu_t = self._max_gpu_temp(gpu_ids)
            cpu_safe = cpu_t is None or cpu_t <= self.max_cpu
            gpu_safe = gpu_t is None or gpu_t <= self.max_gpu
            if cpu_safe and gpu_safe:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"thermal safety timeout: cpu={cpu_t!r}C gpu={gpu_t!r}C "
                    f"limits=({self.max_cpu},{self.max_gpu})"
                )
            time.sleep(2.0)

    @staticmethod
    def _max_cpu_temp() -> float | None:
        try:
            temps = psutil.sensors_temperatures()
        except Exception:
            return None
        values = [entry.current for entries in temps.values() for entry in entries if entry.current]
        return max(values) if values else None

    @staticmethod
    def _max_gpu_temp(gpu_ids: list[int]) -> float | None:
        if pynvml is None or not gpu_ids:
            return None
        values: list[float] = []
        try:
            pynvml.nvmlInit()
            for gpu_id in gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                values.append(float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)))
            pynvml.nvmlShutdown()
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return None
        return max(values) if values else None


class ExperimentRunner:
    def __init__(
        self,
        worker_path: Path,
        config: RootConfig,
        topology: SystemTopologyModel,
        dataset_factory: DatasetFactory,
        results: ResultsStore,
    ) -> None:
        self.worker_path = worker_path
        self.config = config
        self.topology = topology
        self.dataset_factory = dataset_factory
        self.results = results
        self.validator = ResultValidator()
        self.thermal = ThermalGuard(topology, config)
        self.telemetry = TelemetryCollector(config.telemetry)

    def run_task(self, task: TaskSpec, sequence_index: int) -> None:
        started_monotonic = time.monotonic()
        started_iso = utc_now_iso()
        task_base = self._task_metadata(task, sequence_index)
        self._capture_telemetry(task, sequence_index, "task_entry")
        try:
            self._validate_dataset_spec_memory_budget(task.dataset)
            self._validate_exact_gpu_memory_budget(task)
            dataset = self.dataset_factory.get_or_create(task.dataset)
            self._validate_memory_budget(dataset)
            self.thermal.wait_until_safe(task.gpu_ids)
            self._validate_runtime_idleness(task)
            # `pre_task` is deliberately after the thermal/idleness safety gates; `task_entry`
            # preserves the state before any wait so cooldown/order effects remain auditable.
            self._capture_telemetry(task, sequence_index, "pre_task")
            self._run_worker(task, dataset, sequence_index, started_iso)
        except KeyboardInterrupt:
            self._capture_telemetry(task, sequence_index, "post_task_interrupted")
            self.results.append_task(
                {
                    **task_base,
                    "status": "interrupted",
                    "error": "KeyboardInterrupt",
                    "timestamp_start": started_iso,
                    "timestamp_end": utc_now_iso(),
                    "duration_s": time.monotonic() - started_monotonic,
                }
            )
            raise
        except Exception as exc:
            self._capture_telemetry(task, sequence_index, "post_task_failed")
            error_text = f"{type(exc).__name__}: {exc}"
            self.results.append_task(
                {
                    **task_base,
                    "status": "failed",
                    "error": error_text,
                    "timestamp_start": started_iso,
                    "timestamp_end": utc_now_iso(),
                    "duration_s": time.monotonic() - started_monotonic,
                }
            )
            print(
                f"FAILED task={task.task_instance_id} algorithm={task.algorithm.id} "
                f"N={task.dataset.size} dtype={task.dataset.dtype.value} "
                f"GPUs={task.gpu_ids} params={task.algorithm_params}: {error_text}",
                file=sys.stderr,
                flush=True,
            )
            return

    def _validate_runtime_idleness(self, task: TaskSpec) -> None:
        if not self.config.measurement.strict_preflight:
            return

        load = psutil.cpu_percent(interval=0.25)
        limit = self.config.measurement.max_preflight_cpu_load_percent
        if load > limit:
            raise RuntimeError(
                f"strict runtime idleness gate: CPU utilization {load:.1f}% exceeds {limit:.1f}%"
            )

        if pynvml is None or not task.gpu_ids:
            return
        try:
            pynvml.nvmlInit()
            for gpu_id in task.gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                compute = []
                try:
                    compute = [int(p.pid) for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                except Exception:
                    pass
                if compute:
                    raise RuntimeError(
                        f"strict runtime idleness gate: GPU {gpu_id} has foreign compute processes {sorted(set(compute))}"
                    )

                if not self.config.measurement.allow_gpu_graphics_processes:
                    graphics_fn = getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None)
                    graphics = []
                    if graphics_fn is not None:
                        try:
                            graphics = [int(p.pid) for p in graphics_fn(handle)]
                        except Exception:
                            pass
                    if graphics:
                        raise RuntimeError(
                            f"strict runtime idleness gate: GPU {gpu_id} has graphics processes {sorted(set(graphics))}"
                        )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run_worker(
        self,
        task: TaskSpec,
        dataset: DatasetArtifact,
        sequence_index: int,
        task_started_iso: str,
    ) -> None:
        command = self._worker_command(task, dataset)
        stderr_path = self.results.stderr_path(task.task_instance_id)
        env = os.environ.copy()
        env["OMP_PROC_BIND"] = "spread"
        env["OMP_PLACES"] = "threads"
        task_monotonic_start = time.monotonic()
        with stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                ready = self._read_worker_event(process.stdout)
                if ready.event == "unsupported":
                    self.results.append_task(
                        {
                            **self._task_metadata(task, sequence_index),
                            "status": "skipped",
                            "reason": ready.payload.get("reason"),
                            "timestamp_start": task_started_iso,
                            "timestamp_end": utc_now_iso(),
                            "duration_s": time.monotonic() - task_monotonic_start,
                        }
                    )
                    process.wait(timeout=10)
                    return
                if ready.event != "ready":
                    raise ProtocolError(f"expected ready, received {ready.event}: {ready.payload}")

                timing_probe_mean_us: float | None = None
                timing_probe_wall_us: float | None = None
                if self.config.measurement.timing_repetitions == "auto":
                    probe_reps = self.config.measurement.timing_probe_repetitions
                    send_command(process.stdin, f"PROBE {probe_reps}")
                    probe = self._read_worker_event(process.stdout)
                    if probe.event != "probe_done":
                        raise ProtocolError(f"expected probe_done, received {probe.event}: {probe.payload}")
                    timing_probe_mean_us = float(probe.payload.get("mean_iteration_us", 0.0))
                    timing_probe_wall_us = float(probe.payload.get("batch_wall_us", 0.0))
                    timing_repetitions = self._choose_timing_repetitions(timing_probe_mean_us)
                else:
                    timing_repetitions = int(self.config.measurement.timing_repetitions)

                print(
                    f"    timing_batch={timing_repetitions} reps"
                    + (f" (probe_mean={timing_probe_mean_us:.3f} us)" if timing_probe_mean_us is not None else ""),
                    flush=True,
                )
                if self.config.telemetry.capture_pre_post_timing:
                    self._capture_telemetry(task, sequence_index, "pre_timing")
                timing_timestamp_start = utc_now_iso()
                send_command(process.stdin, f"TIMING {timing_repetitions}")
                timing_done = self._read_worker_event(process.stdout)
                timing_timestamp_end = utc_now_iso()
                if timing_done.event != "timing_done":
                    raise ProtocolError(
                        f"expected timing_done, received {timing_done.event}: {timing_done.payload}"
                    )
                if self.config.telemetry.capture_pre_post_timing:
                    self._capture_telemetry(task, sequence_index, "post_timing")

                # CPU package energy is meaningful for all strategies when requested (it
                # includes host/control/idle package cost). GPU energy only exists when the
                # task actually owns GPUs. This avoids useless ENERGY batches for CPU-only
                # tasks when only GPU energy was enabled.
                cpu_energy_enabled = self.config.energy.enable_cpu
                gpu_energy_enabled = self.config.energy.enable_gpu and bool(task.gpu_ids)
                energy_enabled = cpu_energy_enabled or gpu_energy_enabled
                energy_repetitions = 0
                energy_metrics: dict[str, Any] | None = None
                measured = None
                energy_validation = None
                energy_timestamp_start: str | None = None
                energy_timestamp_end: str | None = None
                if energy_enabled:
                    timing_estimate_us = float(
                        timing_done.payload.get(
                            "mean_iteration_us",
                            timing_probe_mean_us if timing_probe_mean_us is not None else ready.payload.get("warmup_median_us", 0.0),
                        )
                    )
                    energy_repetitions = self._choose_energy_repetitions(timing_estimate_us)
                    estimated_energy_s = energy_repetitions * max(0.0, timing_estimate_us) / 1_000_000.0
                    print(
                        f"    timing_mean={timing_estimate_us:.3f} us; "
                        f"energy_batch={energy_repetitions} reps (~{estimated_energy_s:.2f}s)",
                        flush=True,
                    )
                    energy = CompositeEnergyMeter(
                        task.gpu_ids,
                        enable_cpu=cpu_energy_enabled,
                        enable_gpu=gpu_energy_enabled,
                        gpu_poll_ms=self.config.energy.gpu_power_fallback_poll_ms,
                    )
                    energy_started = False
                    energy_stopped = False
                    try:
                        if self.config.telemetry.capture_pre_post_energy:
                            self._capture_telemetry(task, sequence_index, "pre_energy")
                        energy_timestamp_start = utc_now_iso()
                        energy.start()
                        energy_started = True
                        send_command(process.stdin, f"ENERGY {energy_repetitions}")
                        energy_timeout_s = min(
                            self.config.measurement.worker_event_timeout_s,
                            max(60.0, estimated_energy_s * 5.0 + 30.0),
                        )
                        measured = self._read_worker_event(process.stdout, timeout_s=energy_timeout_s)
                        if measured.event != "measure_done":
                            raise ProtocolError(
                                f"expected measure_done, received {measured.event}: {measured.payload}"
                            )
                        energy_metrics = energy.stop()
                        energy_stopped = True
                        energy_timestamp_end = utc_now_iso()
                        if self.config.telemetry.capture_pre_post_energy:
                            self._capture_telemetry(task, sequence_index, "post_energy")
                        print(
                            f"    energy batch complete in {float(measured.payload.get('batch_wall_us', 0.0)) / 1e6:.2f}s",
                            flush=True,
                        )
                    finally:
                        if energy_started and not energy_stopped:
                            try:
                                energy.stop()
                            except Exception:
                                pass
                        energy.close()

                    energy_validation = self.validator.validate(
                        actual=measured.payload["result"],
                        reference=dataset.reference_for(task.operation),
                        sum_abs=float(dataset.metadata["sum_abs"]),
                        dtype=task.dataset.dtype,
                        count=task.dataset.size,
                        operation=task.operation,
                    )

                send_command(process.stdin, "DUMP")
                repetitions_seen = 0
                numerical_mismatches = int(energy_validation is not None and not energy_validation.is_correct)
                fatal_numerical_mismatch = bool(
                    energy_validation is not None
                    and not energy_validation.is_correct
                    and self._numerical_mismatch_is_fatal(task)
                )
                while True:
                    event = self._read_worker_event(process.stdout)
                    if event.event == "result":
                        repetitions_seen += 1
                        actual = event.payload["result"]
                        validation = self.validator.validate(
                            actual=actual,
                            reference=dataset.reference_for(task.operation),
                            sum_abs=float(dataset.metadata["sum_abs"]),
                            dtype=task.dataset.dtype,
                            count=task.dataset.size,
                            operation=task.operation,
                        )
                        if not validation.is_correct:
                            numerical_mismatches += 1
                            fatal_numerical_mismatch |= self._numerical_mismatch_is_fatal(task)
                        cpu_elements = int(event.payload.get("cpu_elements", 0) or 0)
                        gpu_elements = int(event.payload.get("gpu_elements_total", 0) or 0)
                        processed = cpu_elements + gpu_elements
                        self.results.append_repetition(
                            {
                                **self._task_metadata(task, sequence_index),
                                **event.payload,
                                "status": "ok",
                                "reference": dataset.reference_for(task.operation),
                                "is_correct": validation.is_correct,
                                "absolute_error": validation.absolute_error,
                                "relative_error": validation.relative_error,
                                "validation_tolerance": validation.tolerance,
                                "dataset_sha256": dataset.metadata["sha256"],
                                "accumulator_semantics": "native_input_dtype",
                                "cpu_work_fraction": (cpu_elements / processed) if processed else 0.0,
                                "gpu_work_fraction": (gpu_elements / processed) if processed else 0.0,
                                "processed_elements": processed,
                            }
                        )
                    elif event.event == "done":
                        break
                    elif event.event == "error":
                        raise RuntimeError(event.payload.get("message", "worker reported an error"))
                    else:
                        raise ProtocolError(f"unexpected worker event: {event.event}")

                if repetitions_seen != timing_repetitions:
                    raise ProtocolError(
                        f"worker returned {repetitions_seen} result rows, expected {timing_repetitions}"
                    )
                if energy_enabled:
                    assert energy_metrics is not None
                    assert measured is not None
                    assert energy_validation is not None
                    batch_energy = self._energy_record(
                        task,
                        sequence_index,
                        energy_repetitions,
                        energy_metrics,
                        measured.payload,
                        energy_validation,
                        energy_timestamp_start,
                        energy_timestamp_end,
                        cpu_energy_enabled,
                        gpu_energy_enabled,
                    )
                    self.results.append_energy_batch(batch_energy)

                self._capture_telemetry(task, sequence_index, "post_task")
                self.results.append_task(
                    {
                        **self._task_metadata(task, sequence_index),
                        "status": "invalid" if fatal_numerical_mismatch else "ok",
                        "numerical_mismatch_count": numerical_mismatches,
                        "numerical_validation_passed": numerical_mismatches == 0,
                        "timestamp_start": task_started_iso,
                        "timestamp_end": utc_now_iso(),
                        "duration_s": time.monotonic() - task_monotonic_start,
                        "timing_timestamp_start": timing_timestamp_start,
                        "timing_timestamp_end": timing_timestamp_end,
                        "energy_timestamp_start": energy_timestamp_start,
                        "energy_timestamp_end": energy_timestamp_end,
                        "timing_repetitions": timing_repetitions,
                        "timing_repetitions_mode": "auto" if self.config.measurement.timing_repetitions == "auto" else "fixed",
                        "timing_probe_mean_us": timing_probe_mean_us,
                        "timing_probe_wall_us": timing_probe_wall_us,
                        "energy_batch_repetitions": energy_repetitions,
                        "warmup_median_us": ready.payload.get("warmup_median_us"),
                        "strategy_create_us": ready.payload.get("strategy_create_us"),
                        "prepare_us": ready.payload.get("prepare_us"),
                        "dataset_replica_count": ready.payload.get("dataset_replica_count"),
                        "dataset_resident_bytes": ready.payload.get("dataset_resident_bytes"),
                        "prepare_metrics": ready.payload.get("prepare_metrics", {}),
                        "timing_batch_wall_us": timing_done.payload.get("batch_wall_us"),
                        "energy_batch_wall_us": measured.payload.get("batch_wall_us") if measured else None,
                        "dataset_sha256": dataset.metadata["sha256"],
                        "accumulator_semantics": "native_input_dtype",
                    }
                )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)

    @staticmethod
    def _numerical_mismatch_is_fatal(task: TaskSpec) -> bool:
        # A tolerance breach means the implementation did not satisfy the benchmark's
        # correctness contract. Floating-point SUM remains order-sensitive, but that is
        # already reflected in ResultValidator's dtype/count/cancellation-aware tolerance.
        # Invalid numerical results must never remain eligible for a performance ranking.
        return True

    def _capture_telemetry(self, task: TaskSpec, sequence_index: int, phase: str) -> None:
        if not self.config.telemetry.enabled:
            return
        cpu_ids = sorted(set(task.cpu_affinity + task.gpu_worker_cpus))
        snapshot = self.telemetry.snapshot(phase, cpu_ids, task.gpu_ids)
        self.results.append_telemetry(
            {
                **self._task_metadata(task, sequence_index),
                **snapshot,
            }
        )

    def _read_worker_event(self, stream: Any, timeout_s: float | None = None):
        return read_event(
            stream,
            timeout_s=self.config.measurement.worker_event_timeout_s if timeout_s is None else timeout_s,
        )

    def _choose_timing_repetitions(self, iteration_us: float) -> int:
        m = self.config.measurement
        if isinstance(m.timing_repetitions, int):
            return m.timing_repetitions
        if iteration_us <= 0:
            return m.timing_min_repetitions
        estimated = math.ceil(m.timing_target_batch_seconds * 1_000_000.0 / iteration_us)
        return max(m.timing_min_repetitions, min(m.timing_max_repetitions, estimated))

    def _choose_energy_repetitions(self, iteration_us: float) -> int:
        m = self.config.measurement
        if isinstance(m.energy_batch_repetitions, int):
            return m.energy_batch_repetitions
        if iteration_us <= 0:
            return m.energy_min_repetitions
        estimated = math.ceil(m.energy_target_batch_seconds * 1_000_000.0 / iteration_us)
        return max(m.energy_min_repetitions, min(m.energy_max_repetitions, estimated))

    def _worker_command(self, task: TaskSpec, dataset: DatasetArtifact) -> list[str]:
        params = {
            "block_size": 256,
            "chunk_size": 1_048_576,
            "min_chunk_size": 65_536,
            "max_chunk_size": 16_777_216,
            "guided_factor": 2.0,
            "target_chunk_ms": 5.0,
            "ema_alpha": 0.25,
            "pipeline_streams": 4,
            "pipeline_chunks": 16,
            "pipeline_chunk_elements": 0,
            **task.algorithm_params,
        }
        cmd = [
            str(self.worker_path),
            "--dataset", str(dataset.data_path),
            "--dtype", task.dataset.dtype.value,
            "--operation", task.operation.value,
            "--count", str(task.dataset.size),
            "--scheduler", task.algorithm.scheduler,
            "--cpu-backend", task.algorithm.cpu_backend or "none",
            "--gpu-backend", task.algorithm.gpu_backend or "none",
            "--transfer-policy", task.algorithm.transfer_policy or "sync",
            "--gpus", ",".join(map(str, task.gpu_ids)),
            "--cpu-affinity", ",".join(map(str, task.cpu_affinity)),
            "--gpu-worker-cpus", ",".join(map(str, task.gpu_worker_cpus)),
            "--cpu-threads", str(max(1, len(task.cpu_affinity))),
            "--warmup-runs", str(self.config.measurement.warmup_runs),
            "--calibration-repetitions", str(self.config.measurement.scheduler_calibration_repetitions),
            "--cache-rotation-target-bytes", str(self.config.measurement.cache_rotation_target_bytes),
            "--cache-rotation-max-replicas", str(self.config.measurement.cache_rotation_max_replicas),
            "--block-size", str(params["block_size"]),
            "--chunk-size", str(params["chunk_size"]),
            "--min-chunk-size", str(params["min_chunk_size"]),
            "--max-chunk-size", str(params["max_chunk_size"]),
            "--guided-factor", str(params["guided_factor"]),
            "--target-chunk-ms", str(params["target_chunk_ms"]),
            "--ema-alpha", str(params["ema_alpha"]),
            "--pipeline-streams", str(params["pipeline_streams"]),
            "--pipeline-chunks", str(params["pipeline_chunks"]),
            "--pipeline-chunk-elements", str(params["pipeline_chunk_elements"]),
        ]
        if task.memory_policy == "interleave" and shutil.which("numactl") and self.topology.numa_nodes:
            nodes = ",".join(map(str, sorted(self.topology.numa_nodes)))
            return ["numactl", f"--interleave={nodes}", *cmd]
        return cmd

    def _dataset_resident_bytes(self, dataset_bytes: int) -> int:
        m = self.config.measurement
        return cache_rotated_resident_bytes(
            dataset_bytes, m.cache_rotation_target_bytes, m.cache_rotation_max_replicas
        )

    def _validate_dataset_spec_memory_budget(self, spec) -> None:
        dataset_bytes = dataset_size_bytes(spec)
        resident = self._dataset_resident_bytes(dataset_bytes)
        fraction = resident / max(1, self.topology.total_ram_bytes)
        if fraction > self.config.measurement.max_dataset_ram_fraction:
            raise MemoryError(
                f"cache-rotated worker dataset would use {fraction:.1%} of RAM "
                f"({resident} bytes resident from {dataset_bytes} source bytes), exceeding configured limit "
                f"{self.config.measurement.max_dataset_ram_fraction:.1%}"
            )

    def _validate_exact_gpu_memory_budget(self, task: TaskSpec) -> None:
        for row in gpu_capacity_rows(
            task, self.topology, self.config.measurement.gpu_memory_safety_fraction
        ):
            if row["estimate_kind"] == "exact" and not row["within_safe_budget"]:
                raise MemoryError(
                    f"GPU {row['gpu_id']} estimated input allocation {row['estimated_input_bytes']} bytes "
                    f"exceeds safe free-VRAM budget {row['safe_budget_bytes']} bytes for {row['algorithm_id']}"
                )

    def _validate_memory_budget(self, dataset: DatasetArtifact) -> None:
        source_bytes = int(dataset.metadata["size_bytes"])
        resident = self._dataset_resident_bytes(source_bytes)
        fraction = resident / max(1, self.topology.total_ram_bytes)
        if fraction > self.config.measurement.max_dataset_ram_fraction:
            raise MemoryError(
                f"cache-rotated worker dataset uses {fraction:.1%} of RAM "
                f"({resident} bytes resident from {source_bytes} source bytes), exceeding configured limit "
                f"{self.config.measurement.max_dataset_ram_fraction:.1%}"
            )

    def _task_metadata(self, task: TaskSpec, sequence_index: int) -> dict[str, Any]:
        return {
            "task_key": task.task_key,
            "task_instance_id": task.task_instance_id,
            "sequence_index": sequence_index,
            "block_index": task.block_index,
            "group_id": task.group_id,
            "algorithm_id": task.algorithm.id,
            "algorithm_role": task.algorithm.role,
            "algorithm_params": task.algorithm_params,
            "dataset_size": task.dataset.size,
            "dtype": task.dataset.dtype.value,
            "operation": task.operation.value,
            "distribution": task.dataset.distribution.value,
            "dataset_seed": task.dataset.seed,
            "gpu_ids": task.gpu_ids,
            "cpu_core_class": task.cpu_core_class,
            "cpu_thread_policy": task.cpu_thread_policy,
            "cpu_numa_node": task.cpu_numa_node,
            "cpu_affinity": task.cpu_affinity,
            "gpu_worker_cpus": task.gpu_worker_cpus,
            "gpu_control_bindings": task.gpu_control_bindings,
            "gpu_control_mode": task.gpu_control_mode,
            "memory_policy": task.memory_policy,
        }

    def _energy_record(
        self,
        task: TaskSpec,
        sequence_index: int,
        repetitions: int,
        metrics: dict[str, Any],
        worker_batch: dict[str, Any],
        validation: Any,
        timestamp_start: str | None,
        timestamp_end: str | None,
        cpu_requested: bool,
        gpu_requested: bool,
    ) -> dict[str, Any]:
        cpu = metrics.get("cpu_energy_j")
        gpu = metrics.get("gpu_energy_j")
        available = [float(x) for x in (cpu, gpu) if x is not None]
        measured = sum(available) if available else None
        measured_parts: list[str] = []
        if cpu is not None:
            measured_parts.append("cpu_package")
        if gpu is not None:
            measured_parts.append("gpu")
        coverage = "+".join(measured_parts) if measured_parts else "none"
        requested_parts = [name for name, flag in (("cpu_package", cpu_requested), ("gpu", gpu_requested)) if flag]
        return {
            **self._task_metadata(task, sequence_index),
            "status": "ok",
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "repetitions": repetitions,
            "cpu_package_energy_j": cpu,
            "gpu_energy_j": gpu,
            "measured_component_energy_j": measured,
            "cpu_package_energy_per_reduction_j": (float(cpu) / repetitions) if cpu is not None else None,
            "gpu_energy_per_reduction_j": (float(gpu) / repetitions) if gpu is not None else None,
            "measured_component_energy_per_reduction_j": (measured / repetitions) if measured is not None else None,
            # Backward-compatible alias, explicitly qualified by energy_coverage.
            "total_energy_per_reduction_j": (measured / repetitions) if measured is not None else None,
            "energy_coverage": coverage,
            "energy_requested_components": requested_parts,
            "energy_complete_for_requested_components": set(measured_parts) == set(requested_parts),
            "energy_window_s": metrics.get("measurement_window_s"),
            "cpu_energy_method": metrics.get("cpu_energy_method"),
            "cpu_energy_domains_j": metrics.get("cpu_energy_domains_j"),
            "gpu_energy_devices": metrics.get("gpu_energy_devices"),
            "worker_batch_wall_us": worker_batch.get("batch_wall_us"),
            "energy_batch_result": worker_batch.get("result"),
            "energy_batch_is_correct": validation.is_correct,
            "energy_batch_absolute_error": validation.absolute_error,
            "energy_batch_validation_tolerance": validation.tolerance,
        }
