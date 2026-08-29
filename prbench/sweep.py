from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from .catalog import AlgorithmCatalog
from .models import AlgorithmDefinition, DatasetSpec, ExperimentGroup, ReductionOperation, RootConfig, SystemTopologyModel
from .topology import resolve_cpu_pool
from .utils import stable_hash


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    task_instance_id: str
    block_index: int
    group_id: str
    algorithm: AlgorithmDefinition
    algorithm_params: dict[str, Any]
    dataset: DatasetSpec
    operation: ReductionOperation
    gpu_ids: list[int]
    cpu_core_class: str
    cpu_thread_policy: str
    cpu_numa_node: int | None
    cpu_affinity: list[int]
    gpu_worker_cpus: list[int]
    gpu_control_bindings: list[dict[str, Any]]
    gpu_control_mode: str
    memory_policy: str


class SweepPlanner:
    def __init__(self, catalog: AlgorithmCatalog, topology: SystemTopologyModel) -> None:
        self.catalog = catalog
        self.topology = topology

    def plan(self, config: RootConfig) -> list[TaskSpec]:
        base: list[TaskSpec] = []
        for group in config.experiments:
            base.extend(self._plan_group(group))

        all_instances: list[TaskSpec] = []
        for block in range(config.measurement.blocks):
            cloned: list[TaskSpec] = []
            for item in base:
                instance_id = stable_hash({"task_key": item.task_key, "block": block}, 20)
                cloned.append(
                    TaskSpec(
                        **{**item.__dict__, "task_instance_id": instance_id, "block_index": block}
                    )
                )
            rng = random.Random(config.measurement.randomization_seed + block)
            rng.shuffle(cloned)
            all_instances.extend(cloned)
        return all_instances

    def _plan_group(self, group: ExperimentGroup) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        for dataset in group.datasets:
            for operation in group.operations:
                for request in group.algorithms:
                    definition = self.catalog.get(request.id)
                    gpu_sets = self._resolve_gpu_sets(group, definition)
                    params_grid = self._expand_params(request.params)
                    for gpu_ids in gpu_sets:
                        cpu_pool = resolve_cpu_pool(
                            self.topology,
                            group.hardware.cpu_core_class.value,
                            group.hardware.cpu_thread_policy.value,
                            group.hardware.cpu_numa_node,
                            group.hardware.cpu_explicit_ids,
                        )
                        gpu_worker_cpus: list[int] = []
                        gpu_control_bindings: list[dict[str, Any]] = []
                        if definition.uses_gpu:
                            cpu_pool, gpu_worker_cpus, gpu_control_bindings = self._assign_gpu_control_cpus(
                                cpu_pool,
                                gpu_ids,
                                group.hardware.gpu_control_mode.value,
                            )
                        if definition.uses_cpu and not cpu_pool:
                            raise ValueError(f"algorithm {definition.id} has no CPU cores available")
                        for params in params_grid:
                            identity = {
                                "group": group.id,
                                "algorithm": definition.id,
                                "params": params,
                                "dataset": dataset.model_dump(mode="json"),
                                "operation": operation.value,
                                "gpus": gpu_ids,
                                "cpu_core_class": group.hardware.cpu_core_class.value,
                                "cpu_thread_policy": group.hardware.cpu_thread_policy.value,
                                "cpu_numa_node": group.hardware.cpu_numa_node,
                                "cpu_explicit_ids": group.hardware.cpu_explicit_ids,
                                "cpu_affinity": cpu_pool if definition.uses_cpu else [],
                                "gpu_worker_cpus": gpu_worker_cpus,
                                "gpu_control_mode": group.hardware.gpu_control_mode.value,
                                "memory_policy": group.hardware.memory_policy,
                            }
                            tasks.append(
                                TaskSpec(
                                    task_key=stable_hash(identity, 20),
                                    task_instance_id="",
                                    block_index=-1,
                                    group_id=group.id,
                                    algorithm=definition,
                                    algorithm_params=params,
                                    dataset=dataset,
                                    operation=operation,
                                    gpu_ids=gpu_ids,
                                    cpu_core_class=group.hardware.cpu_core_class.value,
                                    cpu_thread_policy=group.hardware.cpu_thread_policy.value,
                                    cpu_numa_node=group.hardware.cpu_numa_node,
                                    cpu_affinity=cpu_pool if definition.uses_cpu else [],
                                    gpu_worker_cpus=gpu_worker_cpus,
                                    gpu_control_bindings=gpu_control_bindings,
                                    gpu_control_mode=group.hardware.gpu_control_mode.value,
                                    memory_policy=group.hardware.memory_policy,
                                )
                            )
        return tasks

    def _resolve_gpu_sets(self, group: ExperimentGroup, definition: AlgorithmDefinition) -> list[list[int]]:
        if not definition.uses_gpu:
            return [[]]
        available = [gpu.index for gpu in self.topology.gpus]
        if not available:
            return []
        resolved: list[list[int]] = []
        for raw in group.hardware.gpu_sets:
            if raw == "each":
                resolved.extend([[gpu] for gpu in available])
            elif raw in {"pairs", "all_pairs"}:
                if not definition.supports_multi_gpu:
                    raise ValueError(f"algorithm {definition.id} does not support gpu_sets={raw!r}")
                resolved.extend([list(pair) for pair in itertools.combinations(available, 2)])
            elif raw == "all":
                if definition.supports_multi_gpu:
                    resolved.append(list(available))
                else:
                    resolved.extend([[gpu] for gpu in available])
            elif isinstance(raw, int):
                resolved.append([raw])
            elif isinstance(raw, list):
                resolved.append([int(x) for x in raw])
            else:
                raise ValueError(f"unsupported gpu_sets entry: {raw!r}")
        unique: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for gpu_set in resolved:
            key = tuple(gpu_set)
            if key in seen:
                continue
            if any(g not in available for g in gpu_set):
                raise ValueError(f"GPU set {gpu_set} contains unavailable device; available={available}")
            if len(gpu_set) < definition.min_gpus:
                continue
            if not definition.supports_multi_gpu and len(gpu_set) != 1:
                raise ValueError(f"algorithm {definition.id} requires exactly one GPU")
            seen.add(key)
            unique.append(gpu_set)
        return unique

    @staticmethod
    def _expand_params(params: dict[str, Any]) -> list[dict[str, Any]]:
        if not params:
            return [{}]

        keys = list(params)
        is_sweep = any(isinstance(params[key], list) for key in keys)
        values = [params[key] if isinstance(params[key], list) else [params[key]] for key in keys]

        valid: list[dict[str, Any]] = []
        invalid_reasons: set[str] = set()
        for product_values in itertools.product(*values):
            combo = dict(zip(keys, product_values))
            reason: str | None = None
            if (
                "min_chunk_size" in combo
                and "max_chunk_size" in combo
                and int(combo["min_chunk_size"]) > int(combo["max_chunk_size"])
            ):
                reason = "min_chunk_size cannot exceed max_chunk_size"
            elif (
                "pipeline_streams" in combo
                and "pipeline_chunks" in combo
                and int(combo["pipeline_chunks"]) < int(combo["pipeline_streams"])
            ):
                reason = "pipeline_chunks cannot be smaller than pipeline_streams"

            if reason is None:
                valid.append(combo)
                continue

            invalid_reasons.add(reason)
            # A scalar configuration represents one explicit request and should fail
            # fast.  For a Cartesian sweep, however, an invalid pair is simply not
            # part of the legal parameter domain (e.g. streams=8, chunks=4).
            if not is_sweep:
                raise ValueError(reason)

        if not valid:
            detail = "; ".join(sorted(invalid_reasons)) or "no valid parameter combinations"
            raise ValueError(f"parameter sweep contains no valid combinations: {detail}")
        return valid

    def _assign_gpu_control_cpus(
        self,
        cpu_pool: list[int],
        gpu_ids: list[int],
        mode: str,
    ) -> tuple[list[int], list[int], list[dict[str, Any]]]:
        """Assign exactly one distinct physical control core to each GPU.

        Local PCI/NUMA CPUs are preferred.  A physical core is never used as the control
        core for two GPUs, even when SMT exposes multiple logical CPUs.  In `dedicated`
        mode every logical CPU belonging to the selected control core is removed from the
        CPU compute pool.  In `shared` mode the core remains in the compute pool and the
        control thread intentionally contends with CPU reduction work.
        """
        if not gpu_ids:
            return list(cpu_pool), [], []
        if mode not in {"dedicated", "shared"}:
            raise ValueError(f"unsupported gpu_control_mode={mode!r}")

        allowed = set(self.topology.allowed_cpus)
        cpu_by_id = {cpu.cpu_id: cpu for cpu in self.topology.logical_cpus}
        selected_compute = set(cpu_pool)
        used_control_cores: set[tuple[int, int]] = set()
        workers: list[int] = []
        bindings: list[dict[str, Any]] = []
        remaining = list(cpu_pool)

        def core_key(cpu_id: int) -> tuple[int, int] | None:
            cpu = cpu_by_id.get(cpu_id)
            return (cpu.socket_id, cpu.core_id) if cpu is not None else None

        for gpu_id in gpu_ids:
            gpu = next((item for item in self.topology.gpus if item.index == gpu_id), None)
            if gpu is None:
                raise ValueError(f"GPU {gpu_id} is unavailable")

            # Candidate tiers encode a deterministic topology-aware preference order.
            # The selected CPU class is respected first, so an E-core-only experiment does
            # not silently use a P-core as its GPU control resource (or vice versa).
            local = [c for c in gpu.local_cpus if c in allowed]
            numa_local = [
                c.cpu_id
                for c in self.topology.logical_cpus
                if c.cpu_id in allowed and gpu.numa_node is not None and c.numa_node == gpu.numa_node
            ]
            selected = [c for c in sorted(selected_compute) if c in allowed]
            tiers: list[tuple[str, list[int]]] = [
                ("pci_local_selected", [c for c in local if c in selected_compute]),
                ("numa_local_selected", [c for c in numa_local if c in selected_compute]),
                ("selected_pool", selected),
                ("pci_local_fallback", local),
                ("numa_local_fallback", numa_local),
                ("global_fallback", sorted(allowed)),
            ]

            chosen: int | None = None
            locality = "fallback"
            for tier_name, candidates in tiers:
                for candidate in candidates:
                    key = core_key(candidate)
                    if key is None or key in used_control_cores:
                        continue
                    chosen = candidate
                    locality = tier_name
                    break
                if chosen is not None:
                    break
            if chosen is None:
                raise ValueError(
                    "not enough distinct physical CPU cores to assign one control core per GPU"
                )

            key = core_key(chosen)
            assert key is not None
            used_control_cores.add(key)
            workers.append(chosen)
            cpu = cpu_by_id[chosen]
            bindings.append(
                {
                    "gpu_id": gpu_id,
                    "cpu_id": chosen,
                    "socket_id": cpu.socket_id,
                    "core_id": cpu.core_id,
                    "numa_node": cpu.numa_node,
                    "gpu_numa_node": gpu.numa_node,
                    "locality": locality,
                    "mode": mode,
                }
            )

            if mode == "dedicated":
                # Remove the complete physical core, not only one SMT sibling.
                remaining = [c for c in remaining if core_key(c) != key]

        return remaining, workers, bindings
