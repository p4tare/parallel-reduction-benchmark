from __future__ import annotations

import math
from typing import Any

from .models import DType, SystemTopologyModel


_DTYPE_BYTES = {
    DType.int32.value: 4,
    DType.float32.value: 4,
    DType.int64.value: 8,
    DType.float64.value: 8,
}


def dtype_size_bytes(dtype: DType | str) -> int:
    key = dtype.value if isinstance(dtype, DType) else str(dtype)
    try:
        return _DTYPE_BYTES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype for capacity estimate: {dtype!r}") from exc


def dataset_size_bytes(dataset: Any) -> int:
    return int(dataset.size) * dtype_size_bytes(dataset.dtype)


def cache_rotation_replicas(dataset_bytes: int, target_bytes: int, max_replicas: int) -> int:
    if dataset_bytes <= 0 or target_bytes <= 0:
        return 1
    desired = math.ceil(target_bytes / dataset_bytes)
    return max(1, min(max_replicas, desired))


def cache_rotated_resident_bytes(dataset_bytes: int, target_bytes: int, max_replicas: int) -> int:
    return dataset_bytes * cache_rotation_replicas(dataset_bytes, target_bytes, max_replicas)


def _param(task: Any, name: str, default: int) -> int:
    value = task.algorithm_params.get(name, default)
    return int(value)


def gpu_input_allocation_estimates(task: Any) -> dict[int, dict[str, Any]]:
    """Estimate dominant per-GPU input allocations for preflight.

    The estimate follows the physical memory path, not merely the scheduler. In
    particular, mapped/HMM input is host memory, Managed Memory is migration-backed,
    and file-stream/chunked execution allocates only bounded device chunks.
    """
    if not task.gpu_ids:
        return {}
    n = int(task.dataset.size)
    item = dtype_size_bytes(task.dataset.dtype)
    g = len(task.gpu_ids)
    scheduler = str(task.algorithm.scheduler)
    transfer = str(task.algorithm.transfer_policy or "sync")
    memory_path = str(task.algorithm.memory_path or "default")
    storage = str(task.algorithm.storage_policy)

    # GPU reads ordinary/mapped/managed host/system memory rather than allocating a
    # full explicit input buffer. CUB scratch/output remains covered by safety headroom.
    if memory_path in {"zero_copy", "hmm_system", "managed_fault", "managed_prefetch", "managed_advised"}:
        return {
            gpu: {"input_bytes": 0, "kind": "managed_or_host_mapped", "elements": n}
            for gpu in task.gpu_ids
        }

    if storage in {"file_stream", "gds"} or memory_path in {"chunked_sync", "pinned_direct"}:
        chunk = _param(task, "pipeline_chunk_elements", 0) if transfer == "async_pipeline" else 0
        if chunk <= 0:
            chunk = _param(task, "chunk_size", 1 << 20)
        streams = _param(task, "pipeline_streams", 4) if transfer == "async_pipeline" else 1
        # Ranges are disjoint, but every GPU owns the same bounded pipeline capacity.
        elements = min(n, chunk) * streams
        return {
            gpu: {"input_bytes": elements * item, "kind": "exact", "elements": elements}
            for gpu in task.gpu_ids
        }

    if scheduler == "gpu_only":
        if transfer == "async_pipeline":
            streams = _param(task, "pipeline_streams", 4)
            chunk_elements = _param(task, "pipeline_chunk_elements", 0)
            if chunk_elements > 0:
                elements = min(n, chunk_elements) * streams
            else:
                chunks = _param(task, "pipeline_chunks", 16)
                elements = math.ceil(n / chunks) * streams
        else:
            elements = n
        return {gpu: {"input_bytes": elements * item, "kind": "exact", "elements": elements} for gpu in task.gpu_ids}

    if scheduler == "gpu_static_equal":
        elements = math.ceil(n / g)
        if transfer == "async_pipeline":
            streams = _param(task, "pipeline_streams", 4)
            chunk_elements = _param(task, "pipeline_chunk_elements", 0)
            if chunk_elements > 0:
                elements = min(elements, chunk_elements) * streams
            else:
                chunks = _param(task, "pipeline_chunks", 16)
                elements = math.ceil(elements / chunks) * streams
        return {gpu: {"input_bytes": elements * item, "kind": "exact", "elements": elements} for gpu in task.gpu_ids}

    if scheduler == "gpu_static_profiled":
        elements = math.ceil(n / g)
        return {gpu: {"input_bytes": elements * item, "kind": "reference", "elements": elements} for gpu in task.gpu_ids}

    if scheduler == "static_equal":
        elements = math.ceil(n / (g + 1))
        if transfer == "async_pipeline":
            streams = _param(task, "pipeline_streams", 4)
            chunk_elements = _param(task, "pipeline_chunk_elements", 0)
            if chunk_elements > 0:
                capacity = min(elements, chunk_elements) * streams
            else:
                chunks = _param(task, "pipeline_chunks", 16)
                capacity = math.ceil(elements / chunks) * streams
            return {gpu: {"input_bytes": capacity * item, "kind": "exact", "elements": capacity} for gpu in task.gpu_ids}
        return {gpu: {"input_bytes": elements * item, "kind": "exact", "elements": elements} for gpu in task.gpu_ids}

    if scheduler == "static_profiled":
        if transfer == "async_pipeline":
            streams = _param(task, "pipeline_streams", 4)
            chunk_elements = _param(task, "pipeline_chunk_elements", 0)
            if chunk_elements > 0:
                capacity = min(n, chunk_elements) * streams
            else:
                chunks = _param(task, "pipeline_chunks", 16)
                capacity = math.ceil(n / chunks) * streams
            return {gpu: {"input_bytes": capacity * item, "kind": "upper_bound", "elements": capacity} for gpu in task.gpu_ids}
        elements = math.ceil(n / (g + 1))
        return {gpu: {"input_bytes": elements * item, "kind": "reference", "elements": elements} for gpu in task.gpu_ids}

    if scheduler == "dynamic_fixed":
        elements = min(n, _param(task, "chunk_size", 1 << 20))
        return {gpu: {"input_bytes": elements * item, "kind": "exact", "elements": elements} for gpu in task.gpu_ids}

    if scheduler in {"dynamic_guided", "dynamic_adaptive"}:
        elements = min(n, _param(task, "max_chunk_size", 1 << 24))
        return {gpu: {"input_bytes": elements * item, "kind": "exact", "elements": elements} for gpu in task.gpu_ids}

    return {gpu: {"input_bytes": n * item, "kind": "reference", "elements": n} for gpu in task.gpu_ids}


def gpu_capacity_rows(task: Any, topology: SystemTopologyModel, safety_fraction: float) -> list[dict[str, Any]]:
    by_id = {g.index: g for g in topology.gpus}
    rows: list[dict[str, Any]] = []
    for gpu_id, estimate in gpu_input_allocation_estimates(task).items():
        gpu = by_id.get(gpu_id)
        if gpu is None:
            continue
        free = int(gpu.memory_free_bytes if gpu.memory_free_bytes is not None else gpu.memory_bytes)
        safe = int(free * safety_fraction)
        required = int(estimate["input_bytes"])
        rows.append({
            "task_key": task.task_key,
            "algorithm_id": task.algorithm.id,
            "memory_path": task.algorithm.memory_path,
            "storage_policy": task.algorithm.storage_policy,
            "gpu_id": gpu_id,
            "gpu_name": gpu.name,
            "estimate_kind": estimate["kind"],
            "estimated_input_bytes": required,
            "gpu_total_bytes": int(gpu.memory_bytes),
            "gpu_free_bytes": free,
            "safe_budget_bytes": safe,
            "within_safe_budget": required <= safe,
        })
    return rows
