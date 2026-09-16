from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .datasets import DatasetFactory
from .models import DatasetSpec, ReductionOperation
from .validation import ResultValidator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryPathStudyConfig(_Strict):
    output_dir: Path = Path("results/memory_paths")
    dataset_cache_dir: Path = Path(".prbench/datasets")
    worker: Path = Path("build/prbench-memory-path-worker")
    datasets: list[DatasetSpec]
    operations: list[ReductionOperation] = Field(default_factory=lambda: [ReductionOperation.sum])
    modes: list[str] = Field(default_factory=lambda: [
        "explicit_sync",
        "chunked_sync",
        "async_pipeline",
        "zero_copy",
        "managed_fault",
        "managed_prefetch",
        "managed_advised",
        "hmm_system",
    ])
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    reuse_counts: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    chunk_elements: list[int] = Field(default_factory=lambda: [4_194_304, 16_777_216, 67_108_864])
    pipeline_streams: list[int] = Field(default_factory=lambda: [2, 4])
    cuda_graph_modes: list[str] = Field(default_factory=list)
    warmup: int = Field(default=1, ge=0, le=20)
    repetitions: int = Field(default=3, ge=1, le=100)
    blocks: int = Field(default=3, ge=1, le=100)
    randomization_seed: int = 20260916
    max_dataset_ram_fraction: float = Field(default=0.70, gt=0.05, le=0.90)
    include_device_resident_diagnostic: bool = True
    include_gpudirect_storage_probe: bool = True

    @field_validator("gpu_ids", "reuse_counts", "chunk_elements", "pipeline_streams")
    @classmethod
    def _non_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("list cannot be empty")
        if any(v < 0 for v in value):
            raise ValueError("values must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_graph_modes(self) -> "MemoryPathStudyConfig":
        allowed = {"explicit_sync", "device_resident"}
        invalid = sorted(set(self.cuda_graph_modes) - allowed)
        if invalid:
            raise ValueError(f"cuda_graph_modes currently support only {sorted(allowed)}; invalid={invalid}")
        return self


@dataclass(frozen=True)
class Task:
    block: int
    dataset: DatasetSpec
    operation: ReductionOperation
    mode: str
    gpu_id: int
    reuse_count: int
    chunk_elements: int
    streams: int
    use_cuda_graphs: bool = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: Path) -> MemoryPathStudyConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MemoryPathStudyConfig.model_validate(raw)


def gds_diagnostics() -> dict[str, Any]:
    modules = ""
    try:
        modules = Path("/proc/modules").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    gdscheck = shutil.which("gdscheck")
    libcufile = None
    for candidate in (
        "/usr/lib/x86_64-linux-gnu/libcufile.so",
        "/usr/local/cuda/lib64/libcufile.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
    ):
        if Path(candidate).exists():
            libcufile = candidate
            break
    return {
        "nvidia_fs_module_loaded": "nvidia_fs " in modules,
        "nvidia_fs_device": Path("/dev/nvidia-fs").exists(),
        "gdscheck": gdscheck,
        "libcufile": libcufile,
        "available_for_native_adapter": bool(
            libcufile and ("nvidia_fs " in modules or Path("/dev/nvidia-fs").exists())
        ),
        "note": (
            "GDS remains an out-of-core/storage experiment and is not mixed into "
            "host-resident ranking."
        ),
    }


def hmm_diagnostics() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"nvidia_smi": None, "addressing_mode_lines": [], "hmm_reported": False}
    try:
        out = subprocess.run(
            [smi, "-q"], text=True, capture_output=True, timeout=15, check=False
        ).stdout
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "nvidia_smi": smi,
            "error": str(exc),
            "addressing_mode_lines": [],
            "hmm_reported": False,
        }
    lines = [line.strip() for line in out.splitlines() if "Addressing Mode" in line]
    return {
        "nvidia_smi": smi,
        "addressing_mode_lines": lines,
        "hmm_reported": any("HMM" in line for line in lines),
    }


def build_tasks(cfg: MemoryPathStudyConfig) -> list[Task]:
    modes = list(cfg.modes)
    if cfg.include_device_resident_diagnostic and "device_resident" not in modes:
        modes.append("device_resident")
    base: list[Task] = []
    for ds in cfg.datasets:
        for op in cfg.operations:
            for mode in modes:
                for gpu in cfg.gpu_ids:
                    for reuse in cfg.reuse_counts:
                        chunks = (
                            cfg.chunk_elements
                            if mode in {"chunked_sync", "async_pipeline"}
                            else [cfg.chunk_elements[0]]
                        )
                        streams = cfg.pipeline_streams if mode == "async_pipeline" else [1]
                        for chunk in chunks:
                            for stream_count in streams:
                                base.append(
                                    Task(
                                        0,
                                        ds,
                                        op,
                                        mode,
                                        gpu,
                                        reuse,
                                        chunk,
                                        stream_count,
                                        mode in cfg.cuda_graph_modes,
                                    )
                                )
    tasks: list[Task] = []
    for block in range(cfg.blocks):
        cloned = [
            Task(
                block,
                x.dataset,
                x.operation,
                x.mode,
                x.gpu_id,
                x.reuse_count,
                x.chunk_elements,
                x.streams,
                x.use_cuda_graphs,
            )
            for x in base
        ]
        random.Random(cfg.randomization_seed + block).shuffle(cloned)
        tasks.extend(cloned)
    return tasks


def run_worker(
    worker: Path,
    artifact_path: Path,
    task: Task,
    count: int,
    dtype: str,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    cmd = [
        str(worker),
        "--dataset", str(artifact_path),
        "--dtype", dtype,
        "--operation", task.operation.value,
        "--mode", task.mode,
        "--count", str(count),
        "--chunk-elements", str(task.chunk_elements),
        "--streams", str(task.streams),
        "--device", str(task.gpu_id),
        "--reuse-count", str(task.reuse_count),
        "--warmup", str(warmup),
        "--repetitions", str(repetitions),
    ]
    if task.use_cuda_graphs:
        cmd.append("--cuda-graphs")
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        reason = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"exit={completed.returncode}"
        )
        lower = reason.lower()
        skip_tokens = (
            "unavailable",
            "not supported",
            "invalid device",
            "pageablememoryaccess=0",
            "out of memory",
            "memory allocation",
        )
        status = "skipped" if any(k in lower for k in skip_tokens) else "failed"
        return {"status": status, "reason": reason, "command": cmd}
    lines = [x for x in completed.stdout.splitlines() if x.strip()]
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


def _summary(rows: list[dict[str, Any]], path: Path) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("is_correct") is True]
    fields = [
        "dataset_bytes", "dtype", "operation", "mode", "gpu_id", "reuse_count",
        "chunk_elements", "streams", "use_cuda_graphs", "mean_total_ms", "mean_h2d_ms",
        "mean_kernel_ms", "mean_d2h_ms", "mean_h2d_bytes", "mean_remote_host_read_bytes",
        "is_correct", "absolute_error", "relative_error", "block",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ok:
            writer.writerow(row)


def cmd_doctor(args: argparse.Namespace) -> int:
    report = {
        "host_ram_bytes": psutil.virtual_memory().total,
        "hmm": hmm_diagnostics(),
        "gpudirect_storage": gds_diagnostics(),
        "memory_path_worker": str(
            (_project_root() / "build/prbench-memory-path-worker").resolve()
        ),
        "gds_worker": str((_project_root() / "build/prbench-gds-worker").resolve()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = _project_root()
    cfg = load_config(Path(args.config).resolve())
    worker = cfg.worker if cfg.worker.is_absolute() else root / cfg.worker
    if not worker.exists():
        raise SystemExit(
            f"memory path worker not found: {worker}; build the branch with CUDA first"
        )
    output = cfg.output_dir if cfg.output_dir.is_absolute() else root / cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset_factory = DatasetFactory(
        cfg.dataset_cache_dir
        if cfg.dataset_cache_dir.is_absolute()
        else root / cfg.dataset_cache_dir
    )
    validator = ResultValidator()
    ram = psutil.virtual_memory().total
    artifacts: dict[str, Any] = {}
    for ds in cfg.datasets:
        bytes_ = ds.size * {
            "int32": 4,
            "float32": 4,
            "int64": 8,
            "float64": 8,
        }[ds.dtype.value]
        if bytes_ > ram * cfg.max_dataset_ram_fraction:
            raise MemoryError(
                f"dataset {bytes_} B exceeds configured host-RAM fraction "
                f"{cfg.max_dataset_ram_fraction:.1%}"
            )
        key = json.dumps(ds.model_dump(mode="json"), sort_keys=True)
        artifacts[key] = dataset_factory.get_or_create(ds)

    manifest = {
        "schema": 3,
        "config": cfg.model_dump(mode="json"),
        "host_ram_bytes": ram,
        "hmm": hmm_diagnostics(),
        "gpudirect_storage": (
            gds_diagnostics() if cfg.include_gpudirect_storage_probe else None
        ),
        "note": (
            "Host-resident memory-path study. GPUDirect Storage is diagnosed separately "
            "and excluded from this ranking."
        ),
    }
    (output / "memory_path_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, sort_keys=True), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    out_jsonl = output / "memory_path_repetitions.jsonl"
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for index, task in enumerate(build_tasks(cfg)):
            key = json.dumps(task.dataset.model_dump(mode="json"), sort_keys=True)
            artifact = artifacts[key]
            dataset_bytes = int(artifact.metadata["size_bytes"])
            result = run_worker(
                worker,
                artifact.data_path,
                task,
                task.dataset.size,
                task.dataset.dtype.value,
                cfg.warmup,
                cfg.repetitions,
            )
            validation_payload: dict[str, Any] = {}
            status = result.get("status", "failed")
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
                "gpu_id": task.gpu_id,
                "reuse_count": task.reuse_count,
                "chunk_elements": task.chunk_elements,
                "streams": task.streams,
                "use_cuda_graphs": task.use_cuda_graphs,
                **result,
                **validation_payload,
                "status": status,
            }
            rows.append(row)
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
            print(
                f"[{index + 1}] block={task.block} mode={task.mode} gpu={task.gpu_id} "
                f"N={task.dataset.size} reuse={task.reuse_count}: {row['status']}",
                flush=True,
            )
    _summary(rows, output / "memory_path_summary.csv")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="prbench-memory-paths")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="report HMM/GDS prerequisites")
    doctor.set_defaults(func=cmd_doctor)
    run = sub.add_parser("run", help="run host-resident GPU memory-path study")
    run.add_argument("config")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
