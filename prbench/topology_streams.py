from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .datasets import DatasetFactory
from .models import DatasetSpec, ReductionOperation
from .validation import ResultValidator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopologyStreamStudyConfig(_Strict):
    output_dir: Path = Path("results/topology_streams")
    dataset_cache_dir: Path = Path(".prbench/datasets")
    worker: Path = Path("build/prbench-topology-stream-worker")
    datasets: list[DatasetSpec]
    operations: list[ReductionOperation] = Field(default_factory=lambda: [ReductionOperation.sum])
    modes: list[str] = Field(
        default_factory=lambda: ["pinned_direct", "multi_gpu_async", "hybrid_cpu_gpu"]
    )
    storage_policies: list[str] = Field(default_factory=lambda: ["host_resident"])
    gpu_sets: list[list[int]] = Field(default_factory=lambda: [[0], [0, 1]])
    gpu_numa_nodes: dict[int, int] = Field(default_factory=dict)
    chunk_elements: list[int] = Field(default_factory=lambda: [16_777_216])
    pipeline_streams: list[int] = Field(default_factory=lambda: [2])
    cpu_threads: list[int] = Field(default_factory=lambda: [1])
    cpu_fractions: list[float] = Field(default_factory=lambda: [0.25])
    warmup: int = Field(default=1, ge=0, le=20)
    repetitions: int = Field(default=3, ge=1, le=100)
    blocks: int = Field(default=1, ge=1, le=100)
    randomization_seed: int = 20260919
    host_resident_ram_fraction: float = Field(default=0.70, gt=0.05, le=0.95)
    numa_strict: bool = False

    @field_validator("modes")
    @classmethod
    def _valid_modes(cls, values: list[str]) -> list[str]:
        allowed = {"pinned_direct", "multi_gpu_async", "hybrid_cpu_gpu"}
        invalid = sorted(set(values) - allowed)
        if invalid:
            raise ValueError(f"unsupported modes: {invalid}")
        return values

    @field_validator("storage_policies")
    @classmethod
    def _valid_storage(cls, values: list[str]) -> list[str]:
        allowed = {"host_resident", "file_stream"}
        invalid = sorted(set(values) - allowed)
        if invalid:
            raise ValueError(f"unsupported storage policies: {invalid}")
        return values

    @field_validator("gpu_sets")
    @classmethod
    def _gpu_sets_non_empty(cls, values: list[list[int]]) -> list[list[int]]:
        if not values or any(not group for group in values):
            raise ValueError("gpu_sets must contain non-empty GPU groups")
        if any(device < 0 for group in values for device in group):
            raise ValueError("GPU ids must be non-negative")
        return values

    @field_validator("chunk_elements", "pipeline_streams", "cpu_threads")
    @classmethod
    def _positive_lists(cls, values: list[int]) -> list[int]:
        if not values or any(value < 1 for value in values):
            raise ValueError("values must be positive")
        return values

    @field_validator("cpu_fractions")
    @classmethod
    def _fractions(cls, values: list[float]) -> list[float]:
        if not values or any(value < 0.0 or value >= 1.0 for value in values):
            raise ValueError("cpu_fractions must be in [0,1)")
        return values


@dataclass(frozen=True)
class Task:
    block: int
    dataset: DatasetSpec
    operation: ReductionOperation
    mode: str
    storage_policy: str
    gpu_ids: tuple[int, ...]
    chunk_elements: int
    streams: int
    cpu_threads: int
    cpu_fraction: float


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: Path) -> TopologyStreamStudyConfig:
    return TopologyStreamStudyConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def build_tasks(cfg: TopologyStreamStudyConfig) -> list[Task]:
    base: list[Task] = []
    for dataset in cfg.datasets:
        for operation in cfg.operations:
            for mode in cfg.modes:
                for storage in cfg.storage_policies:
                    if mode == "pinned_direct" and storage != "host_resident":
                        continue
                    gpu_sets = [[group[0]] for group in cfg.gpu_sets] if mode == "pinned_direct" else cfg.gpu_sets
                    seen_gpu_sets: set[tuple[int, ...]] = set()
                    for group in gpu_sets:
                        gpu_ids = tuple(group)
                        if gpu_ids in seen_gpu_sets:
                            continue
                        seen_gpu_sets.add(gpu_ids)
                        for chunk in cfg.chunk_elements:
                            for streams in cfg.pipeline_streams:
                                if mode == "hybrid_cpu_gpu":
                                    for threads in cfg.cpu_threads:
                                        for fraction in cfg.cpu_fractions:
                                            base.append(
                                                Task(
                                                    0,
                                                    dataset,
                                                    operation,
                                                    mode,
                                                    storage,
                                                    gpu_ids,
                                                    chunk,
                                                    streams,
                                                    threads,
                                                    fraction,
                                                )
                                            )
                                else:
                                    base.append(
                                        Task(
                                            0,
                                            dataset,
                                            operation,
                                            mode,
                                            storage,
                                            gpu_ids,
                                            chunk,
                                            streams,
                                            cfg.cpu_threads[0],
                                            0.0,
                                        )
                                    )
    tasks: list[Task] = []
    for block in range(cfg.blocks):
        cloned = [
            Task(
                block,
                task.dataset,
                task.operation,
                task.mode,
                task.storage_policy,
                task.gpu_ids,
                task.chunk_elements,
                task.streams,
                task.cpu_threads,
                task.cpu_fraction,
            )
            for task in base
        ]
        random.Random(cfg.randomization_seed + block).shuffle(cloned)
        tasks.extend(cloned)
    return tasks


def _dataset_bytes(dataset: DatasetSpec) -> int:
    return dataset.size * {
        "int32": 4,
        "float32": 4,
        "int64": 8,
        "float64": 8,
    }[dataset.dtype.value]


def _invoke(
    worker: Path,
    data_path: Path,
    task: Task,
    cfg: TopologyStreamStudyConfig,
) -> dict[str, Any]:
    cmd = [
        str(worker),
        "--dataset",
        str(data_path),
        "--dtype",
        task.dataset.dtype.value,
        "--operation",
        task.operation.value,
        "--mode",
        task.mode,
        "--storage-policy",
        task.storage_policy,
        "--count",
        str(task.dataset.size),
        "--chunk-elements",
        str(task.chunk_elements),
        "--streams",
        str(task.streams),
        "--gpu-ids",
        ",".join(str(value) for value in task.gpu_ids),
        "--cpu-threads",
        str(task.cpu_threads),
        "--cpu-fraction",
        str(task.cpu_fraction),
        "--warmup",
        str(cfg.warmup),
        "--repetitions",
        str(cfg.repetitions),
    ]
    nodes = [cfg.gpu_numa_nodes.get(gpu) for gpu in task.gpu_ids]
    if all(node is not None for node in nodes):
        cmd.extend(["--gpu-numa-nodes", ",".join(str(node) for node in nodes)])
        if cfg.numa_strict:
            cmd.append("--numa-strict")

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        reason = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
        lower = reason.lower()
        status = "skipped" if any(
            token in lower
            for token in (
                "unavailable",
                "not supported",
                "invalid device",
                "out of memory",
                "memory allocation",
                "numa-local staging unavailable",
            )
        ) else "failed"
        return {"status": status, "reason": reason, "command": cmd}

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"status": "failed", "reason": "worker produced no JSON", "command": cmd}
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "reason": f"invalid worker JSON: {exc}",
            "stdout": completed.stdout,
            "command": cmd,
        }
    return {"status": "ok", **payload, "command": cmd}


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset_bytes",
        "dtype",
        "operation",
        "mode",
        "storage_policy",
        "gpu_ids",
        "gpu_count",
        "chunk_elements",
        "streams",
        "cpu_threads",
        "cpu_fraction",
        "mean_total_ms",
        "mean_storage_read_ms",
        "mean_host_memcpy_ms",
        "mean_h2d_ms",
        "mean_kernel_ms",
        "mean_d2h_ms",
        "mean_cpu_ms",
        "mean_storage_read_bytes",
        "mean_h2d_bytes",
        "mean_cpu_elements",
        "numa_requested",
        "numa_applied",
        "is_correct",
        "absolute_error",
        "relative_error",
        "block",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if row.get("status") == "ok" and row.get("is_correct") is True:
                writer.writerow(row)


def cmd_run(args: argparse.Namespace) -> int:
    root = _root()
    cfg = load_config(Path(args.config).resolve())
    worker = cfg.worker if cfg.worker.is_absolute() else root / cfg.worker
    if not worker.exists():
        raise SystemExit(f"topology stream worker not found: {worker}; rebuild with CUDA")

    output = cfg.output_dir if cfg.output_dir.is_absolute() else root / cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.dataset_cache_dir if cfg.dataset_cache_dir.is_absolute() else root / cfg.dataset_cache_dir
    factory = DatasetFactory(cache_dir)
    validator = ResultValidator()
    ram = psutil.virtual_memory().total
    host_resident_limit = int(ram * cfg.host_resident_ram_fraction)

    artifacts: dict[str, Any] = {}
    for dataset in cfg.datasets:
        key = json.dumps(dataset.model_dump(mode="json"), sort_keys=True)
        artifacts[key] = factory.get_or_create(dataset)

    manifest = {
        "schema": 1,
        "config": cfg.model_dump(mode="json"),
        "host_ram_bytes": ram,
        "host_resident_limit_bytes": host_resident_limit,
        "note": (
            "Topology-stream study: pinned-direct input, concurrent multi-GPU streaming, "
            "CPU+GPU hybrid streaming, and optional NUMA-local registered staging. "
            "Host-resident and file-stream rows must be analyzed separately."
        ),
    }
    (output / "topology_stream_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    destination = output / "topology_stream_repetitions.jsonl"
    with destination.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(build_tasks(cfg)):
            key = json.dumps(task.dataset.model_dump(mode="json"), sort_keys=True)
            artifact = artifacts[key]
            dataset_bytes = int(artifact.metadata["size_bytes"])

            if task.storage_policy == "host_resident" and dataset_bytes > host_resident_limit:
                result: dict[str, Any] = {
                    "status": "skipped",
                    "reason": "host_resident task exceeds configured host-resident RAM budget",
                }
            else:
                result = _invoke(worker, artifact.data_path, task, cfg)

            status = result.get("status", "failed")
            validation_payload: dict[str, Any] = {}
            if status == "ok" and "result" in result:
                validation = validator.validate(
                    actual=result["result"],
                    reference=artifact.reference_for(task.operation),
                    sum_abs=float(artifact.metadata["sum_abs"]),
                    dtype=task.dataset.dtype,
                    count=task.dataset.size,
                    operation=task.operation,
                )
                validation_payload = {
                    "reference": artifact.reference_for(task.operation),
                    "is_correct": validation.is_correct,
                    "absolute_error": validation.absolute_error,
                    "relative_error": validation.relative_error,
                    "validation_tolerance": validation.tolerance,
                }
                if not validation.is_correct:
                    status = "invalid"

            row = {
                "sequence_index": index,
                "block": task.block,
                "dataset": task.dataset.model_dump(mode="json"),
                "dataset_bytes": dataset_bytes,
                "dataset_sha256": artifact.metadata["sha256"],
                "dtype": task.dataset.dtype.value,
                "operation": task.operation.value,
                "mode": task.mode,
                "storage_policy": task.storage_policy,
                "gpu_ids": list(task.gpu_ids),
                "chunk_elements": task.chunk_elements,
                "streams": task.streams,
                "cpu_threads": task.cpu_threads,
                "cpu_fraction": task.cpu_fraction,
                **result,
                **validation_payload,
                "status": status,
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            print(
                f"[{index + 1}] block={task.block} mode={task.mode} "
                f"storage={task.storage_policy} gpus={list(task.gpu_ids)} "
                f"N={task.dataset.size}: {status}",
                flush=True,
            )

    _write_summary(rows, output / "topology_stream_summary.csv")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="prbench-topology-streams")
    parser.add_argument("config")
    args = parser.parse_args()
    raise SystemExit(cmd_run(args))


if __name__ == "__main__":
    main()
