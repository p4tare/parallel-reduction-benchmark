from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import psutil

from .build import BuildError, CMakeBuilder
from .capacity import cache_rotation_replicas, dataset_size_bytes, gpu_capacity_rows
from .catalog import AlgorithmCatalog
from .config import ConfigurationLoader
from .datasets import DatasetFactory
from .energy import NvmlEnergyMeter, RaplEnergyMeter
from .manifest import create_manifest
from .results import ResultsStore
from .runner import ExperimentRunner
from .sweep import SweepPlanner
from .telemetry import TelemetryCollector
from .topology import SystemTopology, enrich_cpu_core_classes
from .utils import command_output

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _format_bytes(value: int | float) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def cmd_list_algorithms(args: argparse.Namespace) -> int:
    catalog = AlgorithmCatalog()
    for item in catalog.all():
        print(
            f"{item.id:34} | {item.role:30} | CPU={item.uses_cpu!s:5} "
            f"GPU={item.uses_gpu!s:5} | storage={item.storage_policy:13} "
            f"path={item.memory_path or '-':18} | {item.label}"
        )
    return 0


def _gds_diagnostics() -> dict[str, object]:
    modules = ""
    try:
        modules = Path("/proc/modules").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    libcufile = next(
        (
            candidate
            for candidate in (
                "/usr/lib/x86_64-linux-gnu/libcufile.so",
                "/usr/local/cuda/lib64/libcufile.so",
                "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
            )
            if Path(candidate).exists()
        ),
        None,
    )
    return {
        "libcufile": libcufile,
        "gdscheck": shutil.which("gdscheck") or shutil.which("gdscheck.py"),
        "nvidia_fs_module_loaded": "nvidia_fs " in modules,
        "nvidia_fs_device": Path("/dev/nvidia-fs").exists(),
    }


def _hmm_diagnostics() -> dict[str, object]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"addressing_mode_lines": [], "hmm_reported": False}
    try:
        output = command_output([smi, "-q"])
    except Exception:
        output = None
    lines = [
        line.strip()
        for line in (output or "").splitlines()
        if "Addressing Mode" in line
    ]
    return {
        "addressing_mode_lines": lines,
        "hmm_reported": any("HMM" in line for line in lines),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    topology = SystemTopology().discover()
    existing_worker = _project_root() / "build" / "prbench-worker"
    if existing_worker.exists():
        topology = enrich_cpu_core_classes(existing_worker, topology)

    def tool(name: str) -> dict[str, str | None]:
        path = shutil.which(name)
        return {
            "path": str(Path(path).resolve()) if path else None,
            "version": command_output([path, "--version"]) if path else None,
        }

    versioned_gxx = {
        name: tool(name)
        for name in [f"g++-{major}" for major in range(16, 5, -1)]
        if shutil.which(name)
    }
    rapl = RaplEnergyMeter()
    report = {
        "topology": topology.model_dump(mode="json"),
        "rapl_present": Path("/sys/class/powercap/intel-rapl").exists(),
        "rapl_available": rapl.available,
        "rapl_diagnostics": RaplEnergyMeter.diagnostics(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_energy_diagnostics": NvmlEnergyMeter.diagnostics([g.index for g in topology.gpus]),
        "memory_path_capabilities": {
            "hmm": _hmm_diagnostics(),
            "gds": _gds_diagnostics(),
            "libnuma_header": Path("/usr/include/numa.h").exists(),
            "numactl": shutil.which("numactl"),
        },
        "project_root": str(_project_root()),
        "toolchain": {
            "cmake": tool("cmake"),
            "cxx": tool("c++"),
            "gxx": tool("g++"),
            "versioned_gxx": versioned_gxx,
            "nvcc": tool("nvcc"),
        },
        "telemetry_snapshot": TelemetryCollector(type("Cfg", (), {
            "enabled": True,
            "capture_cpu_frequency": True,
            "capture_cpu_temperature": True,
            "capture_gpu_state": True,
        })()).snapshot("doctor", topology.allowed_cpus, [g.index for g in topology.gpus]),
        "environment": {
            "CXX": os.environ.get("CXX"),
            "PRBENCH_CXX_COMPILER": os.environ.get("PRBENCH_CXX_COMPILER"),
            "PRBENCH_CUDA_HOST_COMPILER": os.environ.get("PRBENCH_CUDA_HOST_COMPILER"),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    root = _project_root()
    catalog = AlgorithmCatalog()
    loader = ConfigurationLoader(catalog)
    config = loader.load(Path(args.config))
    topology = SystemTopology().discover()
    artifact = CMakeBuilder(root).build(config.build, topology)
    topology = enrich_cpu_core_classes(artifact.worker_path, topology)
    print(json.dumps({
        "worker": str(artifact.worker_path),
        **artifact.metadata,
        "topology": topology.model_dump(mode="json"),
    }, indent=2, default=str))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = _project_root()
    config_path = Path(args.config).resolve()
    catalog = AlgorithmCatalog()
    config = ConfigurationLoader(catalog).load(config_path)
    topology = SystemTopology().discover()

    artifact = CMakeBuilder(root).build(config.build, topology)
    topology = enrich_cpu_core_classes(artifact.worker_path, topology)
    planner = SweepPlanner(catalog, topology)
    tasks = planner.plan(config)
    if not tasks:
        print("No runnable tasks were generated for the detected hardware.", file=sys.stderr, flush=True)
        return 2

    preflight = _evaluate_preflight(config, topology, tasks)
    if preflight["status"] == "failed":
        print("Preflight failed; benchmark was not started.", file=sys.stderr, flush=True)
        print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr, flush=True)
        return 4
    for warning in preflight.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)

    results = ResultsStore(config.output_dir)
    results.write_config_snapshot(config_path.read_text(encoding="utf-8"))
    results.write_manifest(create_manifest(root, config, topology, artifact.metadata))
    factory = DatasetFactory(config.dataset_cache_dir)

    unique_datasets = {}
    for task in tasks:
        key = task.dataset.model_dump_json()
        unique_datasets.setdefault(key, task.dataset)
    print(f"Preparing {len(unique_datasets)} unique dataset(s) before measurements...", flush=True)
    dataset_manifest: list[dict[str, object]] = []
    for dataset_index, dataset_spec in enumerate(unique_datasets.values(), start=1):
        size_bytes = dataset_size_bytes(dataset_spec)
        cached = factory.cached_artifact(dataset_spec)
        print(
            f"  dataset [{dataset_index}/{len(unique_datasets)}] N={dataset_spec.size} "
            f"dtype={dataset_spec.dtype.value} size={_format_bytes(size_bytes)} "
            f"({'cached' if cached else 'generate'})",
            flush=True,
        )
        last_bucket = {-1}

        def progress(done: int, total: int) -> None:
            if total <= 0:
                return
            percent = int(done * 100 / total)
            bucket = min(100, (percent // 5) * 5)
            if done == total:
                bucket = 100
            if bucket <= last_bucket.pop():
                last_bucket.add(bucket)
                return
            last_bucket.add(bucket)
            written_bytes = int(size_bytes * (done / total))
            print(
                f"    {bucket:3d}%  {_format_bytes(written_bytes)} / {_format_bytes(size_bytes)}",
                flush=True,
            )

        dataset_artifact = factory.get_or_create(dataset_spec, progress=progress)
        dataset_manifest.append({
            "spec": dataset_spec.model_dump(mode="json"),
            "data_path": str(dataset_artifact.data_path),
            "metadata_path": str(dataset_artifact.metadata_path),
            "metadata": dataset_artifact.metadata,
        })
    results.write_dataset_manifest(dataset_manifest)

    runner = ExperimentRunner(artifact.worker_path, config, topology, factory, results)

    print(f"Run directory: {results.run_dir}", flush=True)
    print(f"Planned task instances: {len(tasks)}", flush=True)
    campaign_started = time.monotonic()
    completed_durations: list[float] = []
    for index, task in enumerate(tasks, start=1):
        print(
            f"[{index}/{len(tasks)}] {task.algorithm.id} "
            f"op={task.operation.value} N={task.dataset.size} dtype={task.dataset.dtype.value} "
            f"GPUs={task.gpu_ids} storage={task.algorithm.storage_policy} "
            f"path={task.algorithm.memory_path or '-'} block={task.block_index + 1}/{config.measurement.blocks}",
            flush=True,
        )
        task_started = time.monotonic()
        try:
            runner.run_task(task, index)
        except KeyboardInterrupt:
            results.write_summary()
            results.finalize_manifest(status="interrupted")
            print(
                f"Interrupted by user during task {index}/{len(tasks)}. Partial results remain in {results.run_dir}",
                file=sys.stderr,
                flush=True,
            )
            return 130
        duration = time.monotonic() - task_started
        completed_durations.append(duration)
        elapsed = time.monotonic() - campaign_started
        mean_task = sum(completed_durations) / len(completed_durations)
        eta = mean_task * (len(tasks) - index)
        print(
            f"  completed in {duration:.2f}s | campaign elapsed={_format_duration(elapsed)} "
            f"ETA~{_format_duration(eta)}",
            flush=True,
        )
    results.write_summary()
    counts = results.task_status_counts()
    final_status = "completed_with_errors" if counts.get("failed", 0) or counts.get("invalid", 0) else "completed"
    results.finalize_manifest(status=final_status)
    print(f"Task status counts: {json.dumps(counts, sort_keys=True)}", flush=True)
    if counts.get("failed", 0) or counts.get("invalid", 0):
        problems = results.task_problem_rows()
        print(f"Problem task summary ({len(problems)}):", flush=True)
        for row in problems:
            detail = row.get("error")
            if not detail and row.get("status") == "invalid":
                detail = f"numerical mismatch count={row.get('numerical_mismatch_count', 'unknown')}"
            print(
                "  "
                f"status={row.get('status')} algorithm={row.get('algorithm_id')} "
                f"op={row.get('operation')} N={row.get('dataset_size')} "
                f"dtype={row.get('dtype')} GPUs={row.get('gpu_ids')} "
                f"task={row.get('task_instance_id')} reason={detail or 'unspecified'}",
                flush=True,
            )
    print(
        f"Finished in {_format_duration(time.monotonic() - campaign_started)}. Results: {results.run_dir}",
        flush=True,
    )
    if counts.get("failed", 0) or counts.get("invalid", 0):
        print("One or more task instances failed or produced invalid results.", file=sys.stderr)
        return 3
    return 0


def _gpu_processes(gpu_ids: list[int]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    compute: dict[str, list[int]] = {}
    graphics: dict[str, list[int]] = {}
    if pynvml is None or not gpu_ids:
        return compute, graphics
    try:
        pynvml.nvmlInit()
        for gpu_id in gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            compute_pids: list[int] = []
            graphics_pids: list[int] = []
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                compute_pids = sorted({int(p.pid) for p in processes})
            except Exception:
                pass
            fn = getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None)
            if fn is not None:
                try:
                    processes = fn(handle)
                    graphics_pids = sorted({int(p.pid) for p in processes})
                except Exception:
                    pass
            compute[str(gpu_id)] = compute_pids
            graphics[str(gpu_id)] = graphics_pids
    except Exception:
        return compute, graphics
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return compute, graphics


def _cuda_visible_devices_problem(topology) -> str | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return None
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if not tokens or not all(token.isdigit() for token in tokens):
        return (
            f"CUDA_VISIBLE_DEVICES={raw!r} is non-identity/non-numeric; unset it for prbench "
            "so CUDA and NVML GPU indices are guaranteed to match"
        )
    numeric = [int(x) for x in tokens]
    expected = list(range(len(topology.gpus)))
    if numeric != expected:
        return (
            f"CUDA_VISIBLE_DEVICES={raw!r} masks/reorders physical GPUs while prbench topology "
            f"uses NVML indices {expected}; unset it before a research run"
        )
    return None


def _design_warnings(tasks) -> list[str]:
    from collections import defaultdict

    warnings: list[str] = []
    core_ops: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    control_ops: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    dtype_distributions: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for task in tasks:
        base = (
            task.algorithm.id, task.dataset.size, task.dataset.dtype.value,
            task.dataset.distribution.value, tuple(task.gpu_ids),
            tuple(sorted(task.algorithm_params.items())), task.cpu_thread_policy,
        )
        core_ops[base][task.cpu_core_class].add(task.operation.value)
        if task.gpu_ids:
            control_ops[base + (task.cpu_core_class,)][task.gpu_control_mode].add(task.operation.value)
        dtype_base = (
            task.group_id, task.algorithm.id, task.dataset.size, task.operation.value,
            tuple(task.gpu_ids), tuple(sorted(task.algorithm_params.items())),
            task.cpu_core_class, task.cpu_thread_policy, task.gpu_control_mode,
        )
        dtype_distributions[dtype_base][task.dataset.dtype.value].add(task.dataset.distribution.value)
    for base, mapping in core_ops.items():
        if len(mapping) > 1 and len({tuple(sorted(v)) for v in mapping.values()}) > 1:
            warnings.append(
                f"possible confounding: CPU core classes do not cover the same operations for {base}: "
                f"{dict((k, sorted(v)) for k, v in mapping.items())}"
            )
    for base, mapping in control_ops.items():
        if len(mapping) > 1 and len({tuple(sorted(v)) for v in mapping.values()}) > 1:
            warnings.append(
                f"possible confounding: dedicated/shared GPU-control modes do not cover the same operations for {base}: "
                f"{dict((k, sorted(v)) for k, v in mapping.items())}"
            )
    for base, mapping in dtype_distributions.items():
        if len(mapping) > 1 and len({tuple(sorted(v)) for v in mapping.values()}) > 1:
            warnings.append(
                f"possible dtype confounding: compared dtypes use different data distributions for {base}: "
                f"{dict((k, sorted(v)) for k, v in mapping.items())}"
            )
    return warnings


def _evaluate_preflight(config, topology, tasks) -> dict[str, object]:
    warnings: list[str] = []
    fatal: list[str] = []
    gpu_ids = sorted({gpu for task in tasks for gpu in task.gpu_ids})

    cvd_problem = _cuda_visible_devices_problem(topology)
    if cvd_problem:
        fatal.append(cvd_problem)

    rapl = RaplEnergyMeter() if config.energy.enable_cpu else None
    if config.energy.enable_cpu and (rapl is None or not rapl.available):
        fatal.append("CPU energy is enabled but no readable package-level RAPL counter was found")
    if config.energy.enable_gpu and gpu_ids and not topology.nvml_available:
        fatal.append("GPU energy is enabled but NVML is unavailable")
    gpu_energy_diagnostics = NvmlEnergyMeter.diagnostics(gpu_ids) if config.energy.enable_gpu else []
    if config.measurement.strict_preflight and config.energy.enable_gpu and gpu_ids:
        diagnostics_by_gpu = {int(item.get("gpu_id", -1)): item for item in gpu_energy_diagnostics}
        for gpu_id in gpu_ids:
            item = diagnostics_by_gpu.get(gpu_id, {})
            if not (
                bool(item.get("total_energy_counter_supported"))
                or bool(item.get("power_usage_supported"))
            ):
                fatal.append(
                    f"GPU {gpu_id} has neither a readable total-energy counter nor readable power telemetry; "
                    "strict thesis energy measurement cannot proceed"
                )

    gpu_compute_processes, gpu_graphics_processes = _gpu_processes(gpu_ids)
    busy = {gpu: pids for gpu, pids in gpu_compute_processes.items() if pids}
    if busy:
        fatal.append(f"other GPU compute processes are present: {busy}; use exclusive GPUs/node")
    graphics_busy = {gpu: pids for gpu, pids in gpu_graphics_processes.items() if pids}
    if graphics_busy:
        message = (
            f"GPU graphics processes are present: {graphics_busy}; GPU energy includes their activity. "
            "Prefer non-display GPUs for thesis energy measurements."
        )
        if config.measurement.strict_preflight and not config.measurement.allow_gpu_graphics_processes:
            fatal.append(message)
        else:
            warnings.append(message)

    cpu_load = psutil.cpu_percent(interval=0.5)
    if cpu_load > config.measurement.max_preflight_cpu_load_percent:
        message = (
            f"system-wide CPU utilization is already {cpu_load:.1f}% before the run "
            f"(limit {config.measurement.max_preflight_cpu_load_percent:.1f}%)"
        )
        if config.measurement.strict_preflight:
            fatal.append(message)
        else:
            warnings.append(message)

    if config.measurement.strict_preflight:
        dirty = command_output(["git", "status", "--porcelain"], cwd=_project_root())
        if dirty:
            fatal.append("strict_preflight requires a clean Git working tree; commit/stash local changes first")

    classes: dict[str, list[int]] = {}
    for cpu in topology.logical_cpus:
        if cpu.cpu_id in topology.allowed_cpus:
            classes.setdefault(cpu.core_class, []).append(cpu.cpu_id)
    if any(t.cpu_core_class in {"performance", "efficiency"} for t in tasks):
        if "performance" not in classes or "efficiency" not in classes:
            fatal.append(f"P/E-core experiment requested but native core classification is incomplete: {classes}")

    ram_limit = int(topology.total_ram_bytes * config.measurement.max_dataset_ram_fraction)
    available_ram = int(psutil.virtual_memory().available)
    cache_dir = Path(config.dataset_cache_dir)
    factory = DatasetFactory(cache_dir)
    disk = shutil.disk_usage(cache_dir)
    storage_growth_limit = int(disk.free * config.measurement.max_dataset_storage_fraction_of_free)
    missing_storage_bytes = 0

    dataset_checks: dict[str, dict[str, object]] = {}
    tasks_by_dataset: dict[str, list[object]] = {}
    for task in tasks:
        tasks_by_dataset.setdefault(task.dataset.model_dump_json(), []).append(task)

    for key, related in tasks_by_dataset.items():
        task = related[0]
        size = dataset_size_bytes(task.dataset)
        cached = factory.cached_artifact(task.dataset) is not None
        if not cached:
            missing_storage_bytes += size
        host_required = any(t.algorithm.storage_policy == "host_resident" for t in related)
        target = int(config.measurement.cache_rotation_target_bytes)
        replicas = cache_rotation_replicas(size, target, config.measurement.cache_rotation_max_replicas)
        resident = size * replicas if host_required else 0
        storage_policies = sorted({t.algorithm.storage_policy for t in related})
        item = {
            "dataset_size": task.dataset.size,
            "dtype": task.dataset.dtype.value,
            "size_bytes": size,
            "cached": cached,
            "storage_policies": storage_policies,
            "host_resident_required": host_required,
            "cache_rotation_replicas": replicas if host_required else 0,
            "estimated_worker_resident_bytes": resident,
            "configured_ram_limit_bytes": ram_limit,
            "available_ram_bytes_at_preflight": available_ram,
        }
        dataset_checks[key] = item
        if host_required and resident > ram_limit:
            fatal.append(
                f"host-resident dataset {task.dataset.size}x{task.dataset.dtype.value} requires about {resident} "
                f"resident bytes, exceeding max_dataset_ram_fraction budget {ram_limit}; use a file_stream "
                "algorithm for out-of-core execution"
            )
        elif host_required and resident > int(available_ram * 0.85):
            warnings.append(
                f"host-resident dataset {task.dataset.size}x{task.dataset.dtype.value} consumes >85% of currently "
                "available RAM; page cache/worker/CUDA allocations may cause memory pressure"
            )

    if missing_storage_bytes > storage_growth_limit:
        fatal.append(
            f"missing dataset cache files require about {_format_bytes(missing_storage_bytes)}, but the configured "
            f"storage growth budget is {_format_bytes(storage_growth_limit)} "
            f"({config.measurement.max_dataset_storage_fraction_of_free:.0%} of {_format_bytes(disk.free)} free)"
        )

    gpu_memory_rows: list[dict[str, object]] = []
    seen_memory: set[tuple] = set()
    for task in tasks:
        identity = (task.task_key, tuple(task.gpu_ids))
        if identity in seen_memory:
            continue
        seen_memory.add(identity)
        for row in gpu_capacity_rows(task, topology, config.measurement.gpu_memory_safety_fraction):
            gpu_memory_rows.append(row)
            if row["within_safe_budget"]:
                continue
            message = (
                f"GPU {row['gpu_id']} memory headroom: {row['algorithm_id']} estimates "
                f"{row['estimated_input_bytes']} input bytes vs safe budget {row['safe_budget_bytes']} "
                f"({row['estimate_kind']})"
            )
            if row["estimate_kind"] == "exact":
                fatal.append(message)
            else:
                warnings.append(message + "; final model-based/managed placement may still fit")

    by_gpu = {g.index: g for g in topology.gpus}
    multi_sets = sorted({tuple(t.gpu_ids) for t in tasks if len(t.gpu_ids) > 1})
    for gpu_set in multi_sets:
        selected = [by_gpu[g] for g in gpu_set if g in by_gpu]
        names = {g.name for g in selected}
        ccs = {g.compute_capability for g in selected}
        if len(names) > 1 or len(ccs) > 1:
            warnings.append(
                f"heterogeneous GPU set {list(gpu_set)} detected (names={sorted(names)}, cc={sorted(map(str, ccs))}); "
                "equal partition is a baseline only; prefer profiled partition for performance conclusions"
            )
        nodes = {g.numa_node for g in selected if g.numa_node is not None}
        if len(nodes) > 1:
            warnings.append(
                f"GPU set {list(gpu_set)} spans NUMA nodes {sorted(nodes)}. Integrated registered/file-stream "
                "staging requests per-GPU NUMA placement when libnuma is available; verify numa_applied in results."
            )

    if any(t.algorithm.memory_path == "hmm_system" for t in tasks) and not _hmm_diagnostics()["hmm_reported"]:
        warnings.append("HMM was requested but nvidia-smi does not report Addressing Mode: HMM; those tasks should capability-skip")
    gds = _gds_diagnostics()
    if any(t.algorithm.storage_policy == "gds" for t in tasks) and not gds.get("libcufile"):
        warnings.append("GDS was requested but libcufile is not present; GDS tasks should capability-skip")

    warnings.extend(_design_warnings(tasks))
    if any(
        t.operation.value == "sum"
        and t.dataset.dtype.value == "float32"
        and t.dataset.size >= 2**24
        and t.dataset.distribution.value in {"ones", "uniform"}
        and t.dataset.low >= 0
        for t in tasks
    ):
        warnings.append(
            "large non-negative float32 SUM uses native float32 accumulation; substantial rounding/stagnation "
            "is a possible numerical-quality outcome and will be recorded rather than hidden"
        )

    rapl_zones = [z.name for z in rapl.zones] if rapl and rapl.available else []
    if len(rapl_zones) > 1:
        warnings.append(
            f"multiple package-level RAPL zones are readable ({rapl_zones}); reported CPU energy is their sum. "
            "Reserve the whole node so idle/foreign work on another socket does not contaminate the result."
        )

    return {
        "status": "failed" if fatal else "ok",
        "task_instances": len(tasks),
        "cpu_core_classes": classes,
        "gpu_ids": gpu_ids,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_compute_processes": gpu_compute_processes,
        "gpu_graphics_processes": gpu_graphics_processes,
        "gpu_energy_diagnostics": gpu_energy_diagnostics,
        "gpu_memory_capacity": gpu_memory_rows,
        "multi_gpu_sets": [list(x) for x in multi_sets],
        "rapl_available": bool(rapl and rapl.available),
        "rapl_package_zones": rapl_zones,
        "rapl_diagnostics": RaplEnergyMeter.diagnostics() if config.energy.enable_cpu else [],
        "nvml_available": topology.nvml_available,
        "cpu_load_percent": cpu_load,
        "dataset_capacity": list(dataset_checks.values()),
        "dataset_cache_storage": {
            "path": str(cache_dir.resolve()),
            "free_bytes": disk.free,
            "missing_dataset_bytes": missing_storage_bytes,
            "configured_growth_limit_bytes": storage_growth_limit,
        },
        "memory_path_capabilities": {
            "hmm": _hmm_diagnostics(),
            "gds": gds,
            "libnuma_header": Path("/usr/include/numa.h").exists(),
        },
        "warnings": warnings,
        "fatal": fatal,
        "note": (
            "Host-resident RAM limits and file-stream storage limits are evaluated separately. "
            "For thesis energy measurements reserve the whole node/selected GPUs exclusively."
        ),
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    root = _project_root()
    config_path = Path(args.config).resolve()
    catalog = AlgorithmCatalog()
    config = ConfigurationLoader(catalog).load(config_path)
    topology = SystemTopology().discover()
    artifact = CMakeBuilder(root).build(config.build, topology)
    topology = enrich_cpu_core_classes(artifact.worker_path, topology)
    tasks = SweepPlanner(catalog, topology).plan(config)
    report = _evaluate_preflight(config, topology, tasks)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 4 if report["status"] == "failed" else 0


def cmd_plan(args: argparse.Namespace) -> int:
    root = _project_root()
    config_path = Path(args.config).resolve()
    catalog = AlgorithmCatalog()
    config = ConfigurationLoader(catalog).load(config_path)
    topology = SystemTopology().discover()
    artifact = CMakeBuilder(root).build(config.build, topology)
    topology = enrich_cpu_core_classes(artifact.worker_path, topology)
    tasks = SweepPlanner(catalog, topology).plan(config)
    payload = {
        "topology": topology.model_dump(mode="json"),
        "task_count": len(tasks),
        "tasks": [
            {
                "sequence_index": i,
                "group_id": t.group_id,
                "algorithm_id": t.algorithm.id,
                "memory_path": t.algorithm.memory_path,
                "storage_policy": t.algorithm.storage_policy,
                "dataset_size": t.dataset.size,
                "dtype": t.dataset.dtype.value,
                "operation": t.operation.value,
                "gpu_ids": t.gpu_ids,
                "cpu_core_class": t.cpu_core_class,
                "cpu_thread_policy": t.cpu_thread_policy,
                "cpu_affinity": t.cpu_affinity,
                "gpu_control_mode": t.gpu_control_mode,
                "gpu_control_bindings": t.gpu_control_bindings,
                "params": t.algorithm_params,
                "block_index": t.block_index,
            }
            for i, t in enumerate(tasks, start=1)
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prbench", description="Research-grade CPU/GPU reduction benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-algorithms", help="show the research algorithm catalog")
    p.set_defaults(func=cmd_list_algorithms)

    p = sub.add_parser("doctor", help="probe the current server and print detected capabilities")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("build", help="configure and build the native worker")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("plan", help="build, resolve topology/affinity and print the experiment plan without measuring")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("preflight", help="validate topology, energy access, storage and machine idleness")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("run", help="run an experiment configuration")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = args.func(args)
    except BuildError as exc:
        print(f"BUILD ERROR: {exc}", file=sys.stderr, flush=True)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
