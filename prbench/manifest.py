from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RootConfig, SystemTopologyModel
from .utils import command_output


def _git_metadata(root: Path) -> dict[str, Any]:
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "dirty": bool(command_output(["git", "status", "--porcelain"], cwd=root) or ""),
    }



def _source_tree_hash(root: Path) -> str:
    """Hash research source/config files so runs remain identifiable even before a Git commit."""
    digest = hashlib.sha256()
    included_roots = ["prbench", "native", "algorithms", "configs", "tests", "docs"]
    standalone = ["CMakeLists.txt", "pyproject.toml", "README.md", "ALGORITHM_SELECTION.md"]
    paths: list[Path] = []
    for dirname in included_roots:
        base = root / dirname
        if base.exists():
            paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(root / name for name in standalone if (root / name).is_file())
    for path in sorted(set(paths), key=lambda x: str(x.relative_to(root))):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cpu_frequency_metadata() -> dict[str, Any]:
    governors: set[str] = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            governors.add(path.read_text(encoding="utf-8").strip())
        except OSError:
            pass
    boost_values: dict[str, str] = {}
    for name, path in {
        "cpufreq_boost": Path("/sys/devices/system/cpu/cpufreq/boost"),
        "intel_no_turbo": Path("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    }.items():
        try:
            boost_values[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return {"governors": sorted(governors), **boost_values}




def _parse_dmidecode_memory(text: str | None) -> dict[str, Any]:
    """Parse a conservative subset of SMBIOS type-17 fields.

    DMI does not expose a universal, trustworthy "channel count" field. We therefore
    report populated devices, locators/banks and configured speeds instead of inventing
    a channel number from vendor-specific labels.
    """
    if not text:
        return {"available": False, "modules": [], "populated_modules": 0}
    modules: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    wanted = {
        "Size", "Form Factor", "Locator", "Bank Locator", "Type",
        "Speed", "Configured Memory Speed", "Manufacturer", "Part Number", "Rank",
    }
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() == "Memory Device":
            if current is not None:
                modules.append(current)
            current = {}
            continue
        if current is None or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in wanted:
            current[key.lower().replace(" ", "_")] = value
    if current is not None:
        modules.append(current)
    populated = [m for m in modules if m.get("size") and m.get("size") not in {"No Module Installed", "Unknown", "0 MB"}]
    speeds = sorted({m.get("configured_memory_speed") or m.get("speed") for m in populated if m.get("configured_memory_speed") or m.get("speed")})
    return {
        "available": bool(modules),
        "modules": modules,
        "populated_modules": len(populated),
        "reported_speeds": speeds,
        "note": "SMBIOS locators/banks are preserved. Channel count is not inferred from vendor-specific labels.",
    }

def _memory_hardware_metadata(config: RootConfig) -> dict[str, Any]:
    if not config.system_baselines.capture_memory_inventory:
        return {"enabled": False}
    result: dict[str, Any] = {"enabled": True}
    # These commands are intentionally best-effort; many clusters restrict DMI access.
    dmidecode = command_output(["dmidecode", "--type", "17"])
    result["dmidecode_type17"] = dmidecode
    result["dmidecode_parsed"] = _parse_dmidecode_memory(dmidecode)
    result["lshw_memory"] = command_output(["lshw", "-class", "memory"])
    result["edac"] = []
    for mc in sorted(Path("/sys/devices/system/edac/mc").glob("mc[0-9]*")):
        controller: dict[str, Any] = {"controller": mc.name, "dimms": []}
        for dimm in sorted(mc.glob("dimm*")):
            item: dict[str, Any] = {"dimm": dimm.name}
            for field in ("dimm_label", "dimm_location", "size", "dimm_mem_type", "dimm_dev_type"):
                try:
                    item[field] = (dimm / field).read_text(encoding="utf-8").strip()
                except OSError:
                    pass
            controller["dimms"].append(item)
        result["edac"].append(controller)
    return result


def _stream_baseline(config: RootConfig) -> dict[str, Any] | None:
    executable = config.system_baselines.stream_executable
    if not executable:
        return None
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved:
        return {"requested": executable, "available": False, "error": "executable not found"}
    command = [resolved, *config.system_baselines.stream_args]
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=config.system_baselines.stream_timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"requested": executable, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    parsed: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        match = re.match(r"^\s*(Copy|Scale|Add|Triad):\s+([0-9.]+)", line, flags=re.IGNORECASE)
        if match:
            parsed[f"{match.group(1).lower()}_mb_s"] = float(match.group(2))
    return {
        "requested": executable,
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": command,
        "parsed": parsed,
        "raw_output": proc.stdout.strip(),
    }


def _package_versions() -> dict[str, str]:
    names = ["numpy", "psutil", "PyYAML", "pydantic", "nvidia-ml-py"]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def create_manifest(
    project_root: Path,
    config: RootConfig,
    topology: SystemTopologyModel,
    build_metadata: dict[str, object],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(project_root),
        "source_tree_sha256": _source_tree_hash(project_root),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "packages": _package_versions(),
        },
        "platform": {
            "platform": platform.platform(),
            "hostname": topology.hostname,
        },
        "topology": topology.model_dump(mode="json"),
        "build": build_metadata,
        "config": config.model_dump(mode="json"),
        "environment": {
            "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
            "OMP_PLACES": os.environ.get("OMP_PLACES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_OMP_PROC_BIND": "spread",
            "worker_OMP_PLACES": "threads",
        },
        "nvidia_smi": command_output(["nvidia-smi", "-q"]) if topology.gpus else None,
        "lscpu": command_output(["lscpu"]),
        "cpu_frequency": _cpu_frequency_metadata(),
        "kernel_cmdline": Path("/proc/cmdline").read_text(encoding="utf-8").strip() if Path("/proc/cmdline").exists() else None,
        "numactl_hardware": command_output(["numactl", "--hardware"]),
        "memory_hardware": _memory_hardware_metadata(config),
        "stream_baseline": _stream_baseline(config),
        "nvidia_smi_topology": command_output(["nvidia-smi", "topo", "-m"]) if topology.gpus else None,
        "nvidia_smi_p2p_read": command_output(["nvidia-smi", "topo", "-p2p", "r"]) if len(topology.gpus) > 1 else None,
        "nvidia_smi_p2p_write": command_output(["nvidia-smi", "topo", "-p2p", "w"]) if len(topology.gpus) > 1 else None,
    }
