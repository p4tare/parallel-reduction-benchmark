from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
import time

import psutil
from pathlib import Path

from .build import BuildError, CMakeBuilder
from .catalog import AlgorithmCatalog
from .config import ConfigurationLoader
from .datasets import DatasetFactory
from .capacity import cache_rotation_replicas, dataset_size_bytes, gpu_capacity_rows
from .energy import NvmlEnergyMeter, RaplEnergyMeter
from .manifest import create_manifest
from .results import ResultsStore
from .runner import ExperimentRunner
from .sweep import SweepPlanner
from .topology import SystemTopology, enrich_cpu_core_classes
from .telemetry import TelemetryCollector
from .utils import command_output

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_list_algorithms(args: argparse.Namespace) -> int:
    catalog = AlgorithmCatalog()
    for item in catalog.all():
        print(
            f"{item.id:31} | {item.role:28} | CPU={item.uses_cpu!s:5} "
            f"GPU={item.uses_gpu!s:5} | {item.label}"
        )
    return 0


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
    print(json.dumps({"worker": str(artifact.worker_path), **artifact.metadata, "topology": topology.model_dump(mode="json")}, indent=2, default=str))
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

    # Materialize every unique dataset before the randomized measurement sequence.
    # Lazy generation would heat the CPU and storage path only before the first task that
    # happens to use a dataset, creating an avoidable order-dependent thermal confounder.
    unique_datasets = {}
    for task in tasks:
        key = task.dataset.model_dump_json()
        unique_datasets.setdefault(key, task.dataset)
    print(f"Preparing {len(unique_datasets)} unique dataset(s) before measurements...", flush=True)
    dataset_manifest: list[dict[str, object]] = []
    for dataset_index, dataset_spec in enumerate(unique_datasets.values(), start=1):
        print(
            f"  dataset [{dataset_index}/{len(unique_datasets)}] N={dataset_spec.size} dtype={dataset_spec.dtype.value}",
            flush=True,
        )
        dataset_artifact = factory.get_or_create(dataset_spec)
        dataset_manifest.append(
            {
                "spec": dataset_spec.model_dump(mode="json"),
                "data_path": str(dataset_artifact.data_path),
                "metadata_path": str(dataset_artifact.metadata_path),
                "metadata": dataset_artifact.metadata,
            }
        )
    results.write_dataset_manifest(dataset_manifest)

    runner = ExperimentRunner(artifact.worker_path, config, topology, factory, results)

    print(f"Run directory: {results.run_dir}", flush=True)
    print(f"Planned task instances: {len(tasks)}", flush=True)
    for index, task in enumerate(tasks, start=1):
        print(
            f"[{index}/{len(tasks)}] {task.algorithm.id} "
            f"op={task.operation.value} N={task.dataset.size} dtype={task.dataset.dtype.value} GPUs={task.gpu_ids} "
            f"block={task.block_index + 1}/{config.measurement.blocks}",
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
        print(f"  completed in {time.monotonic() - task_started:.2f}s", flush=True)
    results.write_summary()
    counts = results.task_status_counts()
    final_status = "completed_with_errors" if counts.get("failed", 0) or counts.get("invalid", 0) else "completed"
    results.finalize_manifest(status=final_status)
    print(f"Task status counts: {json.dumps(counts, sort_keys=True)}", flush=True)
    print(f"Finished. Results: {results.run_dir}", flush=True)
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
    """Fail fast when CUDA logical IDs can differ from NVML physical indices.

    The current worker intentionally uses the same integer IDs for CUDA and NVML.
    An identity CUDA_VISIBLE_DEVICES list is safe; masks/reordering would make energy,
    telemetry and CUDA execution refer to different physical devices.
    """
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
    """Detect common confounding patterns across otherwise comparable tasks."""
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

        # When dtype is intended as the compared factor, changing distribution at the
        # same time makes the effect ambiguous. We cannot know the research intent, so
        # this is a warning rather than a fatal error.
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

    # Reject impossible host datasets before the generator creates a huge cache file.
    ram_limit = int(topology.total_ram_bytes * config.measurement.max_dataset_ram_fraction)
    available_ram = int(psutil.virtual_memory().available)
    dataset_checks: dict[tuple, dict[str, object]] = {}
    for task in tasks:
        key = (task.dataset.size, task.dataset.dtype.value, task.dataset.seed, task.dataset.distribution.value)
        if key in dataset_checks:
            continue
        size = dataset_size_bytes(task.dataset)
        target = int(config.measurement.cache_rotation_target_bytes)
        replicas = cache_rotation_replicas(
            size, target, config.measurement.cache_rotation_max_replicas
        )
        resident = size * replicas
        item = {
            "dataset_size": task.dataset.size,
            "dtype": task.dataset.dtype.value,
            "size_bytes": size,
            "cache_rotation_replicas": replicas,
            "estimated_worker_resident_bytes": resident,
            "configured_ram_limit_bytes": ram_limit,
            "available_ram_bytes_at_preflight": available_ram,
        }
        dataset_checks[key] = item
        if resident > ram_limit:
            fatal.append(
                f"dataset {task.dataset.size}x{task.dataset.dtype.value} requires about {resident} resident bytes "
                f"with cache rotation ({replicas} replicas), exceeding max_dataset_ram_fraction budget "
                f"{ram_limit} bytes"
            )
        elif resident > int(available_ram * 0.85):
            warnings.append(
                f"cache-rotated dataset {task.dataset.size}x{task.dataset.dtype.value} consumes >85% of currently "
                "available RAM; page cache/worker/CUDA allocations may cause memory pressure"
            )

    # Device-memory feasibility. Exact estimates can fail-fast; model-based/reference
    # estimates are reported as warnings because the final partition is data-dependent.
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
                warnings.append(message + "; final model-based partition may still fit")

    # Multi-GPU topology checks.
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
                f"GPU set {list(gpu_set)} spans NUMA nodes {sorted(nodes)}. Current host dataset is one shared "
                "allocation; memory_policy=interleave is reproducible but is not per-GPU NUMA-local placement. "
                "Report this as a topology limitation when interpreting multi-GPU H2D scaling."
            )

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
        "warnings": warnings,
        "fatal": fatal,
        "note": (
            "For thesis energy measurements reserve the whole node/selected GPUs exclusively. "
            "RAPL is package-wide; multi-GPU host memory is currently one shared allocation."
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

    p = sub.add_parser("preflight", help="validate topology, energy access and machine idleness before a research run")
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
