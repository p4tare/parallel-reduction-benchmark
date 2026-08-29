from __future__ import annotations

import glob
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RaplZone:
    name: str
    energy_path: Path
    max_range_uj: int


@dataclass
class EnergySnapshot:
    timestamp_ns: int
    rapl_uj: dict[str, int] = field(default_factory=dict)
    gpu_mj: dict[int, int] = field(default_factory=dict)


class RaplEnergyMeter:
    def __init__(self) -> None:
        self.zones = self._discover_zones()

    @staticmethod
    def _discover_zones() -> list[RaplZone]:
        zones: list[RaplZone] = []
        roots = [Path(p) for p in glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*")]
        # Keep package-level zones only; nested domains have an extra ':' in basename.
        for root in roots:
            if root.name.count(":") != 1:
                continue
            energy = root / "energy_uj"
            max_range = root / "max_energy_range_uj"
            name_path = root / "name"
            if not energy.exists() or not max_range.exists():
                continue
            try:
                zone_name = name_path.read_text(encoding="utf-8").strip() if name_path.exists() else root.name
                max_uj = int(max_range.read_text(encoding="utf-8").strip())
                _ = int(energy.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            zones.append(RaplZone(f"{root.name}:{zone_name}", energy, max_uj))
        return zones

    @staticmethod
    def diagnostics() -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        roots = [Path(p) for p in glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*")]
        for root in roots:
            if root.name.count(":") != 1:
                continue
            energy = root / "energy_uj"
            max_range = root / "max_energy_range_uj"
            item: dict[str, Any] = {
                "zone": str(root),
                "energy_path": str(energy),
                "max_energy_range_path": str(max_range),
                "energy_exists": energy.exists(),
                "energy_readable": os.access(energy, os.R_OK),
                "max_range_readable": os.access(max_range, os.R_OK),
            }
            for key, path in (("energy", energy), ("max_range", max_range)):
                if path.exists():
                    try:
                        st = path.stat()
                        item[f"{key}_mode"] = oct(st.st_mode & 0o777)
                        item[f"{key}_uid"] = st.st_uid
                        item[f"{key}_gid"] = st.st_gid
                    except OSError as exc:
                        item[f"{key}_stat_error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        item[f"{key}_sample"] = path.read_text(encoding="utf-8").strip()
                    except OSError as exc:
                        item[f"{key}_read_error"] = f"{type(exc).__name__}: {exc}"
            report.append(item)
        return report

    @property
    def available(self) -> bool:
        return bool(self.zones)

    def read(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for zone in self.zones:
            values[zone.name] = int(zone.energy_path.read_text(encoding="utf-8").strip())
        return values

    def delta_j(self, start: dict[str, int], end: dict[str, int]) -> tuple[float, dict[str, float]]:
        total_uj = 0
        per_zone: dict[str, float] = {}
        by_name = {z.name: z for z in self.zones}
        for name, start_val in start.items():
            if name not in end or name not in by_name:
                continue
            end_val = end[name]
            max_range = by_name[name].max_range_uj
            delta = end_val - start_val if end_val >= start_val else (max_range - start_val) + end_val
            total_uj += delta
            per_zone[name] = delta / 1_000_000.0
        return total_uj / 1_000_000.0, per_zone


class NvmlEnergyMeter:
    """Prefer GPU total-energy counters; use timestamped power integration only as fallback."""

    def __init__(self, gpu_ids: list[int], poll_ms: int = 100) -> None:
        self.gpu_ids = list(gpu_ids)
        self.poll_s = poll_ms / 1000.0
        self.available = False
        self.handles: dict[int, Any] = {}
        self.counter_supported: dict[int, bool] = {}
        self.samples: dict[int, list[tuple[int, float]]] = {i: [] for i in self.gpu_ids}
        self._running = False
        self._thread: threading.Thread | None = None
        if pynvml is None or not self.gpu_ids:
            return
        try:
            pynvml.nvmlInit()
            for gpu_id in self.gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.handles[gpu_id] = handle
                try:
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    self.counter_supported[gpu_id] = True
                except Exception:
                    self.counter_supported[gpu_id] = False
            self.available = bool(self.handles)
        except Exception:
            self.available = False


    @staticmethod
    def diagnostics(gpu_ids: list[int]) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        if pynvml is None:
            return report
        try:
            pynvml.nvmlInit()
            for gpu_id in gpu_ids:
                item: dict[str, Any] = {"gpu_id": gpu_id}
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    item["name"] = str(pynvml.nvmlDeviceGetName(handle))
                    try:
                        item["total_energy_counter_supported"] = True
                        item["total_energy_counter_sample_mj"] = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                    except Exception as exc:
                        item["total_energy_counter_supported"] = False
                        item["total_energy_counter_error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        item["power_usage_supported"] = True
                        item["power_usage_sample_w"] = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    except Exception as exc:
                        item["power_usage_supported"] = False
                        item["power_usage_error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                report.append(item)
        except Exception:
            return report
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return report

    def close(self) -> None:
        if pynvml is not None and self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def start(self) -> tuple[dict[int, int], int]:
        counters: dict[int, int] = {}
        now = time.perf_counter_ns()
        if not self.available:
            return counters, now
        self.samples = {i: [] for i in self.gpu_ids}
        for gpu_id, handle in self.handles.items():
            if self.counter_supported.get(gpu_id, False):
                counters[gpu_id] = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
        if any(not self.counter_supported.get(i, False) for i in self.gpu_ids):
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return counters, now

    def stop(self, start_counters: dict[int, int], start_ns: int) -> dict[str, Any]:
        end_ns = time.perf_counter_ns()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_s * 4))
            self._thread = None

        per_gpu: dict[str, Any] = {}
        total_j = 0.0
        for gpu_id, handle in self.handles.items():
            if self.counter_supported.get(gpu_id, False) and gpu_id in start_counters:
                end_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                delta_mj = end_mj - start_counters[gpu_id]
                # NVML total energy is monotonically increasing on supported devices; protect against reset.
                energy_j = max(0.0, delta_mj / 1000.0)
                method = "nvml_total_energy_counter"
            else:
                energy_j = self._integrate_samples(self.samples.get(gpu_id, []), start_ns, end_ns)
                method = "nvml_power_trapezoid_fallback"
            total_j += energy_j
            per_gpu[str(gpu_id)] = {
                "energy_j": energy_j,
                "method": method,
                "samples": len(self.samples.get(gpu_id, [])),
            }
        return {"total_gpu_energy_j": total_j, "per_gpu": per_gpu, "duration_s": (end_ns - start_ns) / 1e9}

    def _poll_loop(self) -> None:
        while self._running:
            ts = time.perf_counter_ns()
            for gpu_id, handle in self.handles.items():
                if self.counter_supported.get(gpu_id, False):
                    continue
                try:
                    power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    self.samples[gpu_id].append((ts, power_w))
                except Exception:
                    pass
            time.sleep(self.poll_s)

    @staticmethod
    def _integrate_samples(samples: list[tuple[int, float]], start_ns: int, end_ns: int) -> float:
        if not samples:
            return 0.0
        clipped = [(max(start_ns, min(ts, end_ns)), p) for ts, p in samples if start_ns <= ts <= end_ns]
        if len(clipped) == 1:
            return clipped[0][1] * ((end_ns - start_ns) / 1e9)
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(clipped, clipped[1:]):
            energy += 0.5 * (p0 + p1) * ((t1 - t0) / 1e9)
        # Extend endpoints using nearest sample so the integral covers the entire measurement window.
        energy += clipped[0][1] * ((clipped[0][0] - start_ns) / 1e9)
        energy += clipped[-1][1] * ((end_ns - clipped[-1][0]) / 1e9)
        return max(0.0, energy)


class CompositeEnergyMeter:
    def __init__(self, gpu_ids: list[int], enable_cpu: bool, enable_gpu: bool, gpu_poll_ms: int) -> None:
        self.rapl = RaplEnergyMeter() if enable_cpu else None
        self.nvml = NvmlEnergyMeter(gpu_ids, gpu_poll_ms) if enable_gpu else None
        self._rapl_start: dict[str, int] = {}
        self._gpu_start: dict[int, int] = {}
        self._gpu_start_ns = 0
        self._start_ns = 0

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()
        if self.rapl and self.rapl.available:
            self._rapl_start = self.rapl.read()
        if self.nvml and self.nvml.available:
            self._gpu_start, self._gpu_start_ns = self.nvml.start()

    def stop(self) -> dict[str, Any]:
        end_ns = time.perf_counter_ns()
        result: dict[str, Any] = {"measurement_window_s": (end_ns - self._start_ns) / 1e9}
        if self.rapl and self.rapl.available:
            end = self.rapl.read()
            total, per_zone = self.rapl.delta_j(self._rapl_start, end)
            result["cpu_energy_j"] = total
            result["cpu_energy_method"] = "linux_powercap_rapl_counter"
            result["cpu_energy_domains_j"] = per_zone
        else:
            result["cpu_energy_j"] = None
            result["cpu_energy_method"] = "unavailable"
        if self.nvml and self.nvml.available:
            gpu = self.nvml.stop(self._gpu_start, self._gpu_start_ns)
            result["gpu_energy_j"] = gpu["total_gpu_energy_j"]
            result["gpu_energy_devices"] = gpu["per_gpu"]
        else:
            result["gpu_energy_j"] = None
            result["gpu_energy_devices"] = {}
        return result

    def close(self) -> None:
        if self.nvml:
            self.nvml.close()
