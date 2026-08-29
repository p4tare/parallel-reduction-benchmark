from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ResultsStore:
    """Append-only raw result storage plus derived condition/block summaries."""

    def __init__(self, output_base: Path) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_dir = output_base / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "stderr").mkdir()
        self.repetition_path = self.run_dir / "repetitions.jsonl"
        self.energy_path = self.run_dir / "energy_batches.jsonl"
        self.task_path = self.run_dir / "tasks.jsonl"
        self.telemetry_path = self.run_dir / "telemetry.jsonl"
        self.dataset_manifest_path = self.run_dir / "datasets.json"
        self._run_monotonic_start = time.monotonic()

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(value, sort_keys=True, default=str) + "\n")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def finalize_manifest(self, *, status: str = "completed") -> None:
        path = self.run_dir / "manifest.json"
        if not path.exists():
            return
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["timestamp_end"] = datetime.now(timezone.utc).isoformat()
        manifest["run_duration_s"] = time.monotonic() - self._run_monotonic_start
        manifest["run_status"] = status
        manifest["task_status_counts"] = self.task_status_counts()
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def write_config_snapshot(self, raw_text: str) -> None:
        (self.run_dir / "config.snapshot.yaml").write_text(raw_text, encoding="utf-8")

    def write_dataset_manifest(self, datasets: list[dict[str, Any]]) -> None:
        self.dataset_manifest_path.write_text(
            json.dumps({"schema_version": 1, "datasets": datasets}, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def append_task(self, row: dict[str, Any]) -> None:
        self._append_jsonl(self.task_path, row)

    def append_repetition(self, row: dict[str, Any]) -> None:
        self._append_jsonl(self.repetition_path, row)

    def append_energy_batch(self, row: dict[str, Any]) -> None:
        self._append_jsonl(self.energy_path, row)

    def append_telemetry(self, row: dict[str, Any]) -> None:
        self._append_jsonl(self.telemetry_path, row)

    def stderr_path(self, task_instance_id: str) -> Path:
        return self.run_dir / "stderr" / f"{task_instance_id}.log"

    def task_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._read_jsonl(self.task_path):
            status = str(row.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    def task_problem_rows(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self._read_jsonl(self.task_path)
            if row.get("status") in {"failed", "invalid"}
        ]

    @staticmethod
    def _numeric_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            value = row.get(field)
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
        return values

    @classmethod
    def _stats(cls, rows: list[dict[str, Any]], field: str, prefix: str) -> dict[str, Any]:
        values = cls._numeric_values(rows, field)
        if not values:
            return {}
        return {
            f"{prefix}_median": statistics.median(values),
            f"{prefix}_mean": statistics.fmean(values),
            f"{prefix}_p25": cls._quantile(values, 0.25),
            f"{prefix}_p75": cls._quantile(values, 0.75),
            f"{prefix}_min": min(values),
            f"{prefix}_max": max(values),
        }

    def _summary_row(
        self,
        key: str,
        rows: list[dict[str, Any]],
        energies: list[dict[str, Any]],
        task_rows: list[dict[str, Any]],
        *,
        block_index: int | None = None,
    ) -> dict[str, Any]:
        first = rows[0]
        base: dict[str, Any] = {
            "task_key": key,
            "algorithm_id": first.get("algorithm_id"),
            "algorithm_params": json.dumps(first.get("algorithm_params", {}), sort_keys=True),
            "dataset_size": first.get("dataset_size"),
            "dtype": first.get("dtype"),
            "operation": first.get("operation"),
            "distribution": first.get("distribution"),
            "dataset_seed": first.get("dataset_seed"),
            "gpu_ids": json.dumps(first.get("gpu_ids", [])),
            "cpu_core_class": first.get("cpu_core_class"),
            "cpu_thread_policy": first.get("cpu_thread_policy"),
            "gpu_control_mode": first.get("gpu_control_mode"),
            "memory_policy": first.get("memory_policy"),
            "samples": len(rows),
            "correct_fraction": sum(bool(r.get("is_correct")) for r in rows) / len(rows),
        }
        if block_index is not None:
            base["block_index"] = block_index

        metric_fields = {
            "e2e_us": "e2e_us",
            "cpu_compute_us": "cpu_compute_us",
            "gpu_h2d_sum_us": "gpu_h2d_sum_us",
            "gpu_kernel_sum_us": "gpu_kernel_sum_us",
            "gpu_d2h_sum_us": "gpu_d2h_sum_us",
            "gpu_total_max_us": "gpu_total_max_us",
            "scheduler_us": "scheduler_us",
            "merge_us": "merge_us",
            "cpu_elements": "cpu_elements",
            "gpu_elements_total": "gpu_elements_total",
            "cpu_work_fraction": "cpu_work_fraction",
            "gpu_work_fraction": "gpu_work_fraction",
            "absolute_error": "absolute_error",
            "relative_error": "relative_error",
        }
        for field, prefix in metric_fields.items():
            base.update(self._stats(rows, field, prefix))

        if task_rows:
            base["task_instances"] = len(task_rows)
            for field in ("strategy_create_us", "prepare_us", "timing_batch_wall_us", "timing_probe_mean_us"):
                vals = self._numeric_values(task_rows, field)
                if vals:
                    base[f"{field}_median"] = statistics.median(vals)
            reps = self._numeric_values(task_rows, "timing_repetitions")
            if reps:
                base["timing_repetitions_median"] = statistics.median(reps)

        if energies:
            energy_values = self._numeric_values(energies, "measured_component_energy_per_reduction_j")
            if energy_values:
                base["measured_energy_median_j"] = statistics.median(energy_values)
                e2e_median = base.get("e2e_us_median")
                if e2e_median is not None:
                    base["edp_median_j_s"] = base["measured_energy_median_j"] * (float(e2e_median) / 1e6)
            coverages = sorted({str(e.get("energy_coverage", "unknown")) for e in energies})
            base["energy_coverage"] = ";".join(coverages)
            base["energy_batches"] = len(energies)
        else:
            base["energy_coverage"] = "none"
            base["energy_batches"] = 0
        return base

    def write_summary(self) -> None:
        repetitions = self._read_jsonl(self.repetition_path)
        energy_batches = self._read_jsonl(self.energy_path)
        tasks = self._read_jsonl(self.task_path)
        valid_reps = [r for r in repetitions if r.get("status") == "ok"]
        valid_energy = [e for e in energy_batches if e.get("status") == "ok"]
        valid_tasks = [t for t in tasks if t.get("status") in {"ok", "invalid"}]

        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        energy_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        task_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid_reps:
            by_key[row["task_key"]].append(row)
        for row in valid_energy:
            energy_by_key[row["task_key"]].append(row)
        for row in valid_tasks:
            task_by_key[row["task_key"]].append(row)

        summary = [
            self._summary_row(key, rows, energy_by_key.get(key, []), task_by_key.get(key, []))
            for key, rows in by_key.items()
        ]
        self._write_csv(self.run_dir / "summary.csv", summary)

        # A separate per-block summary makes thermal/order drift inspectable without
        # destroying the raw samples or conflating independently randomized blocks.
        reps_by_block: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        energy_by_block: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        tasks_by_block: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in valid_reps:
            reps_by_block[(row["task_key"], int(row["block_index"]))].append(row)
        for row in valid_energy:
            energy_by_block[(row["task_key"], int(row["block_index"]))].append(row)
        for row in valid_tasks:
            tasks_by_block[(row["task_key"], int(row["block_index"]))].append(row)
        block_summary = [
            self._summary_row(key, rows, energy_by_block.get((key, block), []), tasks_by_block.get((key, block), []), block_index=block)
            for (key, block), rows in reps_by_block.items()
        ]
        self._write_csv(self.run_dir / "block_summary.csv", block_summary)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        fraction = pos - lo
        return ordered[lo] * (1 - fraction) + ordered[hi] * fraction
