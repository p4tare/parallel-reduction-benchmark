from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import DType, DatasetSpec, Distribution, QuantizationMode, ReductionOperation


NP_DTYPES: dict[DType, np.dtype[Any]] = {
    DType.int32: np.dtype(np.int32),
    DType.int64: np.dtype(np.int64),
    DType.float32: np.dtype(np.float32),
    DType.float64: np.dtype(np.float64),
}


@dataclass(frozen=True)
class DatasetArtifact:
    data_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    def reference_for(self, operation: ReductionOperation) -> int | float:
        return self.metadata[f"reference_{operation.value}"]


DATASET_SCHEMA_VERSION = 4

def _canonical_spec(spec: DatasetSpec) -> str:
    payload = {"schema_version": DATASET_SCHEMA_VERSION, "spec": spec.model_dump(mode="json")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _quantize(data: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    q = spec.quantization
    if q.mode == QuantizationMode.none:
        return data
    if q.mode == QuantizationMode.decimal:
        return np.round(data, decimals=int(q.digits or 0)).astype(data.dtype, copy=False)
    step = 2.0 ** (-int(q.bits or 0))
    return (np.round(data / step) * step).astype(data.dtype, copy=False)


def _validate_integer_sum_bound(spec: DatasetSpec) -> None:
    if spec.dtype not in {DType.int32, DType.int64}:
        return
    max_abs = max(abs(int(math.floor(spec.low))), abs(int(math.ceil(spec.high))), 1)
    if spec.distribution == Distribution.ones:
        max_abs = 1
    if spec.distribution == Distribution.zeros:
        max_abs = 0
    accumulator_info = np.iinfo(np.int32 if spec.dtype == DType.int32 else np.int64)
    if max_abs * spec.size > accumulator_info.max:
        raise ValueError(
            f"integer dataset could overflow native {spec.dtype.value} accumulator; "
            "reduce size/range or change the benchmark accumulator policy"
        )


class DatasetFactory:
    """Deterministic, cached dataset generation with reference metadata and content hashes."""

    def __init__(self, cache_dir: Path, chunk_elements: int = 8_000_000) -> None:
        self.cache_dir = cache_dir
        self.chunk_elements = chunk_elements
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, spec: DatasetSpec) -> DatasetArtifact:
        _validate_integer_sum_bound(spec)
        key = hashlib.sha256(_canonical_spec(spec).encode("utf-8")).hexdigest()[:20]
        data_path = self.cache_dir / f"dataset_{key}.bin"
        metadata_path = self.cache_dir / f"dataset_{key}.json"
        if data_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_bytes = spec.size * NP_DTYPES[spec.dtype].itemsize
            if (
                data_path.stat().st_size == expected_bytes
                and metadata.get("spec") == spec.model_dump(mode="json")
                and metadata.get("schema_version") == DATASET_SCHEMA_VERSION
                and all(key in metadata for key in ("reference_sum", "reference_min", "reference_max", "sum_abs"))
            ):
                return DatasetArtifact(data_path.resolve(), metadata_path.resolve(), metadata)

        tmp = data_path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()

        rng = np.random.default_rng(spec.seed)
        dtype = NP_DTYPES[spec.dtype]
        sha = hashlib.sha256()
        reference = np.longdouble(0.0)
        sum_abs = np.longdouble(0.0)
        int_reference = 0
        reference_min: int | float | None = None
        reference_max: int | float | None = None
        written = 0

        with tmp.open("wb") as fh:
            while written < spec.size:
                count = min(self.chunk_elements, spec.size - written)
                data = self._generate_chunk(rng, spec, count)
                raw = data.tobytes(order="C")
                fh.write(raw)
                sha.update(raw)

                if spec.dtype in {DType.int32, DType.int64}:
                    # Safe because _validate_integer_sum_bound guarantees int64 range.
                    chunk_sum = int(np.sum(data, dtype=np.int64))
                    int_reference += chunk_sum
                    sum_abs += np.longdouble(np.sum(np.abs(data.astype(np.int64)), dtype=np.int64))
                    chunk_min = int(np.min(data))
                    chunk_max = int(np.max(data))
                else:
                    reference += np.sum(data, dtype=np.longdouble)
                    sum_abs += np.sum(np.abs(data.astype(np.longdouble)), dtype=np.longdouble)
                    chunk_min = float(np.min(data))
                    chunk_max = float(np.max(data))
                reference_min = chunk_min if reference_min is None else min(reference_min, chunk_min)
                reference_max = chunk_max if reference_max is None else max(reference_max, chunk_max)
                written += count

        tmp.replace(data_path)
        reference_value: int | float
        reference_method: str
        if spec.dtype in {DType.int32, DType.int64}:
            reference_value = int_reference
            reference_method = "numpy_int64_exact_with_prevalidated_no_overflow"
        else:
            reference_value = float(reference)
            reference_method = "numpy_longdouble_chunked"

        if reference_min is None or reference_max is None:
            raise RuntimeError("dataset generation produced no values")

        sum_abs_float = float(sum_abs)
        sum_condition_number = None
        if float(reference_value) != 0.0:
            sum_condition_number = sum_abs_float / abs(float(reference_value))

        metadata = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "spec": spec.model_dump(mode="json"),
            "size_bytes": data_path.stat().st_size,
            "sha256": sha.hexdigest(),
            "reference_sum": reference_value,
            "reference_min": reference_min,
            "reference_max": reference_max,
            "sum_abs": sum_abs_float,
            "sum_condition_number": sum_condition_number,
            "reference_method": reference_method,
            "numpy_longdouble_bits": int(np.finfo(np.longdouble).nmant + 1),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return DatasetArtifact(data_path.resolve(), metadata_path.resolve(), metadata)

    def _generate_chunk(
        self, rng: np.random.Generator, spec: DatasetSpec, count: int
    ) -> np.ndarray:
        dtype = NP_DTYPES[spec.dtype]
        if spec.distribution == Distribution.ones:
            return np.ones(count, dtype=dtype)
        if spec.distribution == Distribution.zeros:
            return np.zeros(count, dtype=dtype)
        if spec.distribution == Distribution.integers:
            if spec.dtype not in {DType.int32, DType.int64}:
                raise ValueError("distribution=integers requires an integer dtype")
            return rng.integers(
                int(math.ceil(spec.low)),
                int(math.floor(spec.high)) + 1,
                size=count,
                dtype=dtype,
            )
        if spec.distribution == Distribution.symmetric_uniform:
            if spec.dtype in {DType.int32, DType.int64}:
                raise ValueError("symmetric_uniform is intended for floating-point datasets")
            data = np.empty(count, dtype=dtype)
            pairs = count // 2
            values = rng.uniform(spec.low, spec.high, size=pairs).astype(dtype)
            values = _quantize(values, spec)
            data[0 : 2 * pairs : 2] = values
            data[1 : 2 * pairs : 2] = -values
            if count % 2:
                data[-1] = dtype.type(0)
            return data
        if spec.distribution == Distribution.uniform:
            if spec.dtype in {DType.int32, DType.int64}:
                return rng.integers(
                    int(math.ceil(spec.low)),
                    int(math.floor(spec.high)) + 1,
                    size=count,
                    dtype=dtype,
                )
            data = rng.uniform(spec.low, spec.high, size=count).astype(dtype)
            return _quantize(data, spec)
        raise ValueError(f"unsupported distribution: {spec.distribution}")
