from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from pathlib import Path

import psutil

from .models import SystemTopologyModel, TopologyCpu, TopologyGpu

try:
    import pynvml  # provided by nvidia-ml-py
except ImportError:  # pragma: no cover - platform dependent
    pynvml = None  # type: ignore[assignment]


def _parse_cpu_list(raw: str) -> list[int]:
    result: list[int] = []
    for token in raw.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


def _read_int(path: Path, default: int | None = None) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return default


def _cpu_numa_node(cpu_path: Path) -> int | None:
    for candidate in cpu_path.glob("node[0-9]*"):
        try:
            return int(candidate.name.removeprefix("node"))
        except ValueError:
            continue
    return None


def _normalize_pci_bus_id(raw: str) -> str:
    # NVML commonly returns 00000000:65:00.0 while sysfs uses 0000:65:00.0.
    domain, bus, rest = raw.split(":", 2)
    return f"{int(domain, 16):04x}:{bus.lower()}:{rest.lower()}"


class SystemTopology:
    """Portable topology discovery with Linux sysfs precision and graceful fallbacks."""

    def discover(self) -> SystemTopologyModel:
        cpus = self._discover_cpus()
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [c.cpu_id for c in cpus]
        numa: dict[int, list[int]] = {}
        for cpu in cpus:
            if cpu.numa_node is not None:
                numa.setdefault(cpu.numa_node, []).append(cpu.cpu_id)
        for values in numa.values():
            values.sort()

        gpus, nvml_ok = self._discover_gpus(numa)
        p2p_matrix = self._discover_p2p_matrix(gpus) if nvml_ok else {}
        return SystemTopologyModel(
            hostname=socket.gethostname(),
            os=platform.system(),
            kernel=platform.release(),
            machine=platform.machine(),
            logical_cpus=cpus,
            allowed_cpus=allowed,
            numa_nodes=numa,
            gpus=gpus,
            total_ram_bytes=int(psutil.virtual_memory().total),
            nvml_available=nvml_ok,
            p2p_matrix=p2p_matrix,
        )

    def _discover_cpus(self) -> list[TopologyCpu]:
        logical = psutil.cpu_count(logical=True) or 1
        result: list[TopologyCpu] = []
        if platform.system() != "Linux":
            physical = psutil.cpu_count(logical=False) or logical
            for cpu_id in range(logical):
                result.append(
                    TopologyCpu(cpu_id=cpu_id, socket_id=0, core_id=cpu_id % physical, numa_node=None)
                )
            return result

        for cpu_id in range(logical):
            base = Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
            socket_id = _read_int(base / "topology" / "physical_package_id", 0) or 0
            core_id = _read_int(base / "topology" / "core_id", cpu_id)
            online = _read_int(base / "online", 1) != 0
            result.append(
                TopologyCpu(
                    cpu_id=cpu_id,
                    socket_id=socket_id,
                    core_id=int(core_id if core_id is not None else cpu_id),
                    numa_node=_cpu_numa_node(base),
                    online=online,
                )
            )
        return result

    def _discover_gpus(self, numa: dict[int, list[int]]) -> tuple[list[TopologyGpu], bool]:
        if pynvml is None:
            return [], False
        gpus: list[TopologyGpu] = []
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = str(pynvml.nvmlDeviceGetName(handle))
                uuid = str(pynvml.nvmlDeviceGetUUID(handle))
                pci = pynvml.nvmlDeviceGetPciInfo(handle)
                raw_bus = pci.busId.decode() if isinstance(pci.busId, bytes) else str(pci.busId)
                bus_id = _normalize_pci_bus_id(raw_bus)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                cc: str | None = None
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    cc = f"{major}.{minor}"
                except Exception:
                    pass

                numa_node: int | None = None
                local_cpus: list[int] = []
                if platform.system() == "Linux":
                    pci_dir = Path("/sys/bus/pci/devices") / bus_id
                    node = _read_int(pci_dir / "numa_node", -1)
                    if node is not None and node >= 0:
                        numa_node = node
                    cpulist_path = pci_dir / "local_cpulist"
                    if cpulist_path.exists():
                        try:
                            local_cpus = _parse_cpu_list(cpulist_path.read_text(encoding="utf-8"))
                        except OSError:
                            pass
                    if not local_cpus and numa_node is not None:
                        local_cpus = list(numa.get(numa_node, []))

                current_speed = None
                current_width = None
                max_speed = None
                max_width = None
                if platform.system() == "Linux":
                    pci_dir = Path("/sys/bus/pci/devices") / bus_id
                    for attr, target in (("current_link_speed", "current"), ("max_link_speed", "max")):
                        try:
                            value = (pci_dir / attr).read_text(encoding="utf-8").strip()
                        except OSError:
                            value = None
                        if target == "current":
                            current_speed = value
                        else:
                            max_speed = value
                    current_width = _read_int(pci_dir / "current_link_width")
                    max_width = _read_int(pci_dir / "max_link_width")

                gpus.append(
                    TopologyGpu(
                        index=index,
                        name=name,
                        uuid=uuid,
                        pci_bus_id=bus_id,
                        memory_bytes=int(mem.total),
                        memory_free_bytes=int(mem.free),
                        memory_used_bytes=int(mem.used),
                        compute_capability=cc,
                        numa_node=numa_node,
                        local_cpus=local_cpus,
                        pcie_current_link_speed=current_speed,
                        pcie_current_link_width=current_width,
                        pcie_max_link_speed=max_speed,
                        pcie_max_link_width=max_width,
                    )
                )
            pynvml.nvmlShutdown()
            return gpus, True
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return [], False

    def _discover_p2p_matrix(self, gpus: list[TopologyGpu]) -> dict[str, dict[str, object]]:
        """Best-effort structured NVIDIA P2P capability matrix.

        Missing NVML symbols/drivers are recorded as `unknown`; lack of a P2P
        capability never prevents single-GPU benchmarking.
        """
        matrix: dict[str, dict[str, object]] = {}
        if pynvml is None or not gpus:
            return matrix
        fn = getattr(pynvml, "nvmlDeviceGetP2PStatus", None)
        read_cap = getattr(pynvml, "NVML_P2P_CAPS_INDEX_READ", None)
        write_cap = getattr(pynvml, "NVML_P2P_CAPS_INDEX_WRITE", None)
        atomics_cap = getattr(pynvml, "NVML_P2P_CAPS_INDEX_ATOMICS", None)
        try:
            pynvml.nvmlInit()
            handles = {g.index: pynvml.nvmlDeviceGetHandleByIndex(g.index) for g in gpus}
            for src in gpus:
                row: dict[str, object] = {}
                for dst in gpus:
                    if src.index == dst.index:
                        row[str(dst.index)] = {"read": True, "write": True, "atomics": True, "status": "self"}
                        continue
                    item: dict[str, object] = {"status": "unknown"}
                    if fn is not None:
                        for label, cap in (("read", read_cap), ("write", write_cap), ("atomics", atomics_cap)):
                            if cap is None:
                                item[label] = None
                                continue
                            try:
                                status = fn(handles[src.index], handles[dst.index], cap)
                                item[label] = int(status) == 0
                                item[f"{label}_nvml_status"] = int(status)
                            except Exception:
                                item[label] = None
                        if any(item.get(k) is not None for k in ("read", "write", "atomics")):
                            item["status"] = "queried"
                    row[str(dst.index)] = item
                matrix[str(src.index)] = row
        except Exception:
            return matrix
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return matrix


def choose_one_thread_per_core(topology: SystemTopologyModel, allowed: set[int]) -> list[int]:
    seen: set[tuple[int, int]] = set()
    selected: list[int] = []
    for cpu in sorted(topology.logical_cpus, key=lambda x: x.cpu_id):
        if not cpu.online or cpu.cpu_id not in allowed:
            continue
        key = (cpu.socket_id, cpu.core_id)
        if key not in seen:
            seen.add(key)
            selected.append(cpu.cpu_id)
    return selected


def enrich_cpu_core_classes(worker_path: Path, topology: SystemTopologyModel) -> SystemTopologyModel:
    """Use the native CPUID probe to classify heterogeneous x86 cores.

    Python/sysfs discovery remains architecture-neutral.  On Intel hybrid x86 systems the
    built worker can pin itself to each allowed logical CPU and query CPUID leaf 0x1A, which
    is substantially safer than inferring P/E type from SMT sibling counts.  Unsupported
    platforms remain explicitly `unknown`/`homogeneous`; no heuristic is silently promoted
    to research metadata.
    """
    if not worker_path.exists() or not topology.allowed_cpus:
        return topology
    command = [
        str(worker_path),
        "--probe-cpu-types",
        "--probe-cpus",
        ",".join(map(str, topology.allowed_cpus)),
    ]
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return topology
    if proc.returncode != 0:
        return topology
    payload: dict[str, object] | None = None
    for line in proc.stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("event") == "cpu_types":
            payload = candidate
            break
    if not payload:
        return topology
    mapping: dict[int, tuple[str, str]] = {}
    for item in payload.get("cpus", []):
        if not isinstance(item, dict):
            continue
        try:
            cpu_id = int(item["cpu_id"])
            core_class = str(item["core_class"])
            source = str(item.get("source", "native_cpuid"))
        except (KeyError, TypeError, ValueError):
            continue
        if core_class not in {"performance", "efficiency", "homogeneous", "unknown"}:
            core_class = "unknown"
        mapping[cpu_id] = (core_class, source)
    if not mapping:
        return topology
    cpus = []
    for cpu in topology.logical_cpus:
        core_class, source = mapping.get(cpu.cpu_id, (cpu.core_class, cpu.core_class_source))
        cpus.append(cpu.model_copy(update={"core_class": core_class, "core_class_source": source}))
    return topology.model_copy(update={"logical_cpus": cpus})


def resolve_cpu_pool(
    topology: SystemTopologyModel,
    core_class: str,
    thread_policy: str,
    numa_node: int | None,
    explicit_ids: list[int] | None = None,
) -> list[int]:
    allowed = set(topology.allowed_cpus)
    cpu_by_id = {c.cpu_id: c for c in topology.logical_cpus}

    if explicit_ids is not None:
        unavailable = [c for c in explicit_ids if c not in allowed or c not in cpu_by_id or not cpu_by_id[c].online]
        if unavailable:
            raise ValueError(f"cpu_explicit_ids contains unavailable CPUs: {unavailable}; allowed={sorted(allowed)}")
        return list(explicit_ids)

    candidates = [c for c in topology.logical_cpus if c.online and c.cpu_id in allowed]
    if numa_node is not None:
        candidates = [c for c in candidates if c.numa_node == numa_node]
        if not candidates:
            raise ValueError(f"no allowed CPUs found on NUMA node {numa_node}")

    if core_class in {"performance", "efficiency"}:
        candidates = [c for c in candidates if c.core_class == core_class]
        if not candidates:
            detected: dict[str, list[int]] = {}
            for cpu in topology.logical_cpus:
                if cpu.cpu_id in allowed:
                    detected.setdefault(cpu.core_class, []).append(cpu.cpu_id)
            raise ValueError(
                f"cpu_core_class={core_class} requested but no such CPUs were detected; "
                f"detected={detected}. Use cpu_explicit_ids only after verifying the machine topology."
            )
    elif core_class != "all":
        raise ValueError(f"unsupported cpu_core_class={core_class!r}")

    selected = [c.cpu_id for c in sorted(candidates, key=lambda x: x.cpu_id)]
    if thread_policy == "one_thread_per_core":
        return choose_one_thread_per_core(topology, set(selected))
    if thread_policy == "all_threads":
        return selected
    raise ValueError(f"unsupported cpu_thread_policy={thread_policy!r}")
