from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .datasets import DatasetFactory
from .memory_paths import gds_diagnostics
from .models import DatasetSpec, ReductionOperation
from .validation import ResultValidator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GdsStudyConfig(_Strict):
    output_dir: Path = Path("results/gds")
    dataset_cache_dir: Path = Path(".prbench/datasets")
    worker: Path = Path("build/prbench-gds-worker")
    datasets: list[DatasetSpec]
    operations: list[ReductionOperation] = Field(default_factory=lambda: [ReductionOperation.sum])
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    chunk_elements: list[int] = Field(default_factory=lambda: [16_777_216, 67_108_864])
    blocks: int = Field(default=3, ge=1, le=100)
    randomization_seed: int = 20260918


@dataclass(frozen=True)
class Task:
    block: int
    dataset: DatasetSpec
    operation: ReductionOperation
    gpu_id: int
    chunk_elements: int


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def load(path: Path) -> GdsStudyConfig:
    return GdsStudyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def tasks(cfg: GdsStudyConfig) -> list[Task]:
    base = [
        Task(0, ds, op, gpu, chunk)
        for ds in cfg.datasets
        for op in cfg.operations
        for gpu in cfg.gpu_ids
        for chunk in cfg.chunk_elements
    ]
    result: list[Task] = []
    for block in range(cfg.blocks):
        cloned = [Task(block, t.dataset, t.operation, t.gpu_id, t.chunk_elements) for t in base]
        random.Random(cfg.randomization_seed + block).shuffle(cloned)
        result.extend(cloned)
    return result


def invoke(worker: Path, data: Path, task: Task) -> dict[str, Any]:
    cmd = [
        str(worker),
        "--dataset", str(data),
        "--dtype", task.dataset.dtype.value,
        "--operation", task.operation.value,
        "--count", str(task.dataset.size),
        "--chunk-elements", str(task.chunk_elements),
        "--device", str(task.gpu_id),
        "--repetitions", "1",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {"status": "failed", "reason": proc.stderr.strip() or proc.stdout.strip(), "command": cmd}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"status": "failed", "reason": "GDS worker produced no result", "command": cmd}
    try:
        return {"status": "ok", **json.loads(lines[-1]), "command": cmd}
    except json.JSONDecodeError as exc:
        return {"status": "failed", "reason": f"invalid GDS worker JSON: {exc}", "command": cmd}


def cmd_run(args: argparse.Namespace) -> int:
    root = _root()
    cfg = load(Path(args.config).resolve())
    worker = cfg.worker if cfg.worker.is_absolute() else root / cfg.worker
    diag = gds_diagnostics()
    if not worker.exists():
        raise SystemExit(
            f"GDS worker not built: {worker}. Diagnostics: {json.dumps(diag, sort_keys=True)}"
        )
    out = cfg.output_dir if cfg.output_dir.is_absolute() else root / cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache = cfg.dataset_cache_dir if cfg.dataset_cache_dir.is_absolute() else root / cfg.dataset_cache_dir
    factory = DatasetFactory(cache)
    validator = ResultValidator()
    artifacts: dict[str, Any] = {}
    for ds in cfg.datasets:
        key = json.dumps(ds.model_dump(mode="json"), sort_keys=True)
        artifacts[key] = factory.get_or_create(ds)

    (out / "gds_manifest.json").write_text(
        json.dumps({"schema": 1, "config": cfg.model_dump(mode="json"), "gds": diag}, indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )
    destination = out / "gds_repetitions.jsonl"
    with destination.open("a", encoding="utf-8") as fh:
        for index, task in enumerate(tasks(cfg)):
            key = json.dumps(task.dataset.model_dump(mode="json"), sort_keys=True)
            artifact = artifacts[key]
            result = invoke(worker, artifact.data_path, task)
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
                    "is_correct": validation.is_correct,
                    "absolute_error": validation.absolute_error,
                    "relative_error": validation.relative_error,
                    "validation_tolerance": validation.tolerance,
                    "reference": artifact.reference_for(task.operation),
                }
                if not validation.is_correct:
                    status = "invalid"
            row = {
                "sequence_index": index,
                "block": task.block,
                "dataset": task.dataset.model_dump(mode="json"),
                "dataset_sha256": artifact.metadata["sha256"],
                "dataset_bytes": artifact.metadata["size_bytes"],
                "operation": task.operation.value,
                "gpu_id": task.gpu_id,
                "chunk_elements": task.chunk_elements,
                **result,
                **validation_payload,
                "status": status,
            }
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
            print(
                f"[{index + 1}] GDS block={task.block} gpu={task.gpu_id} "
                f"N={task.dataset.size} chunk={task.chunk_elements}: {status}",
                flush=True,
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="prbench-gds")
    parser.add_argument("config")
    args = parser.parse_args()
    raise SystemExit(cmd_run(args))


if __name__ == "__main__":
    main()
