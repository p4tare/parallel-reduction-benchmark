from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _cpu_frequency_snapshot(cpu_ids: list[int]) -> dict[str, Any]:
    per_cpu: dict[str, dict[str, Any]] = {}
    values_mhz: list[float] = []
    sources: set[str] = set()
    for cpu_id in sorted(set(cpu_ids)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq")
        candidates = [
            # cpuinfo_avg_freq, when supplied by the scaling driver, is the best
            # kernel-exposed approximation of effective recent frequency.
            ("cpuinfo_avg_freq", base / "cpuinfo_avg_freq"),
            ("cpuinfo_cur_freq", base / "cpuinfo_cur_freq"),
            ("scaling_cur_freq", base / "scaling_cur_freq"),
        ]
        value_khz: int | None = None
        source = "unavailable"
        for candidate_source, path in candidates:
            value_khz = _read_int(path)
            if value_khz is not None:
                source = candidate_source
                break
        # Portable fallback for systems whose cpufreq sysfs is hidden by the
        # container/scheduler. psutil is lower provenance than a per-CPU sysfs value,
        # therefore it is used only when all kernel candidates are unavailable.
        if value_khz is None:
            try:
                freqs = psutil.cpu_freq(percpu=True)
                if cpu_id < len(freqs) and freqs[cpu_id] is not None and freqs[cpu_id].current:
                    value_khz = int(float(freqs[cpu_id].current) * 1000.0)
                    source = "psutil_cpu_freq"
            except Exception:
                pass
        item: dict[str, Any] = {"source": source, "frequency_khz": value_khz}
        if value_khz is not None:
            item["frequency_mhz"] = value_khz / 1000.0
            values_mhz.append(value_khz / 1000.0)
            sources.add(source)
        per_cpu[str(cpu_id)] = item
    return {
        "per_cpu": per_cpu,
        "mean_mhz": (sum(values_mhz) / len(values_mhz)) if values_mhz else None,
        "min_mhz": min(values_mhz) if values_mhz else None,
        "max_mhz": max(values_mhz) if values_mhz else None,
        "sources": sorted(sources),
        "note": (
            "cpuinfo_avg_freq is preferred when exposed by the kernel. Otherwise cpuinfo_cur_freq "
            "or scaling_cur_freq is recorded, with psutil as a final fallback. The source is always stored "
            "because these values are not equivalent to APERF/MPERF effective frequency on every platform."
        ),
    }


def _cpu_temperature_snapshot() -> dict[str, Any]:
    sensors: list[dict[str, Any]] = []
    primary_cpu_values: list[float] = []
    fallback_cpu_values: list[float] = []
    primary_cpu_groups = {"coretemp", "k10temp", "zenpower", "cpu_thermal"}
    try:
        raw = psutil.sensors_temperatures()
    except Exception as exc:  # pragma: no cover - platform dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "sensors": []}
    for group, entries in raw.items():
        group_is_primary_cpu = group.lower() in primary_cpu_groups
        group_is_fallback_cpu = group.lower() == "acpitz"
        for entry in entries:
            if entry.current is None:
                continue
            current = float(entry.current)
            if group_is_primary_cpu:
                primary_cpu_values.append(current)
            elif group_is_fallback_cpu:
                fallback_cpu_values.append(current)
            sensors.append(
                {
                    "group": group,
                    "label": entry.label or None,
                    "current_c": current,
                    "is_cpu_sensor": group_is_primary_cpu or group_is_fallback_cpu,
                    "cpu_sensor_priority": (
                        "primary" if group_is_primary_cpu
                        else "fallback" if group_is_fallback_cpu
                        else None
                    ),
                    "high_c": float(entry.high) if entry.high is not None else None,
                    "critical_c": float(entry.critical) if entry.critical is not None else None,
                }
            )
    selected = primary_cpu_values if primary_cpu_values else fallback_cpu_values
    return {
        "available": bool(selected),
        "max_c": max(selected) if selected else None,
        "sensor_scope": "primary_cpu_groups_else_acpitz_fallback",
        "sensors": sensors,
    }


def _nvml_call(fn_name: str, handle: Any, *args: Any) -> Any | None:
    if pynvml is None:
        return None
    fn = getattr(pynvml, fn_name, None)
    if fn is None:
        return None
    try:
        return fn(handle, *args)
    except Exception:
        return None


def _gpu_state_snapshot(gpu_ids: list[int]) -> dict[str, Any]:
    if pynvml is None or not gpu_ids:
        return {"available": False, "devices": {}}
    devices: dict[str, Any] = {}
    try:
        pynvml.nvmlInit()
        for gpu_id in sorted(set(gpu_ids)):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = _nvml_call("nvmlDeviceGetUtilizationRates", handle)
            pstate = _nvml_call("nvmlDeviceGetPerformanceState", handle)
            temperature = _nvml_call("nvmlDeviceGetTemperature", handle, pynvml.NVML_TEMPERATURE_GPU)
            power_mw = _nvml_call("nvmlDeviceGetPowerUsage", handle)
            power_limit_mw = _nvml_call("nvmlDeviceGetPowerManagementLimit", handle)
            graphics = _nvml_call("nvmlDeviceGetClockInfo", handle, pynvml.NVML_CLOCK_GRAPHICS)
            sm = _nvml_call("nvmlDeviceGetClockInfo", handle, pynvml.NVML_CLOCK_SM)
            memory = _nvml_call("nvmlDeviceGetClockInfo", handle, pynvml.NVML_CLOCK_MEM)
            pcie_gen = _nvml_call("nvmlDeviceGetCurrPcieLinkGeneration", handle)
            pcie_width = _nvml_call("nvmlDeviceGetCurrPcieLinkWidth", handle)
            devices[str(gpu_id)] = {
                "temperature_c": float(temperature) if temperature is not None else None,
                "pstate": f"P{int(pstate)}" if pstate is not None else None,
                "graphics_clock_mhz": int(graphics) if graphics is not None else None,
                "sm_clock_mhz": int(sm) if sm is not None else None,
                "memory_clock_mhz": int(memory) if memory is not None else None,
                "power_w": float(power_mw) / 1000.0 if power_mw is not None else None,
                "power_limit_w": float(power_limit_mw) / 1000.0 if power_limit_mw is not None else None,
                "gpu_utilization_percent": int(util.gpu) if util is not None else None,
                "memory_utilization_percent": int(util.memory) if util is not None else None,
                "pcie_current_generation": int(pcie_gen) if pcie_gen is not None else None,
                "pcie_current_width": int(pcie_width) if pcie_width is not None else None,
            }
        return {"available": True, "devices": devices}
    except Exception as exc:  # pragma: no cover - depends on driver/hardware
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "devices": devices}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class TelemetryCollector:
    """Low-intrusion state snapshots taken outside measured benchmark windows."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def snapshot(self, phase: str, cpu_ids: list[int], gpu_ids: list[int]) -> dict[str, Any]:
        if not self.config.enabled:
            return {"timestamp": utc_now_iso(), "phase": phase, "enabled": False}
        row: dict[str, Any] = {"timestamp": utc_now_iso(), "phase": phase, "enabled": True}
        if self.config.capture_cpu_frequency:
            row["cpu_frequency"] = _cpu_frequency_snapshot(cpu_ids)
        if self.config.capture_cpu_temperature:
            row["cpu_temperature"] = _cpu_temperature_snapshot()
        if self.config.capture_gpu_state:
            row["gpu_state"] = _gpu_state_snapshot(gpu_ids)
        return row
