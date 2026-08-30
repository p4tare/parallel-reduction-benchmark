from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DType(str, Enum):
    int32 = "int32"
    int64 = "int64"
    float32 = "float32"
    float64 = "float64"


class ReductionOperation(str, Enum):
    sum = "sum"
    min = "min"
    max = "max"


class Distribution(str, Enum):
    ones = "ones"
    zeros = "zeros"
    uniform = "uniform"
    integers = "integers"
    symmetric_uniform = "symmetric_uniform"


class QuantizationMode(str, Enum):
    none = "none"
    decimal = "decimal"
    binary_fraction = "binary_fraction"


class QuantizationConfig(StrictModel):
    mode: QuantizationMode = QuantizationMode.none
    digits: int | None = Field(default=None, ge=0, le=12)
    bits: int | None = Field(default=None, ge=0, le=24)

    @model_validator(mode="after")
    def validate_parameters(self) -> "QuantizationConfig":
        if self.mode == QuantizationMode.decimal and self.digits is None:
            raise ValueError("decimal quantization requires 'digits'")
        if self.mode == QuantizationMode.binary_fraction and self.bits is None:
            raise ValueError("binary_fraction quantization requires 'bits'")
        return self


class DatasetSpec(StrictModel):
    size: int = Field(gt=0)
    dtype: DType
    distribution: Distribution = Distribution.uniform
    seed: int = 20260813
    low: float = -1.0
    high: float = 1.0
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetSpec":
        if self.low >= self.high and self.distribution in {
            Distribution.uniform,
            Distribution.integers,
            Distribution.symmetric_uniform,
        }:
            raise ValueError("dataset.low must be smaller than dataset.high")
        if self.dtype in {DType.int32, DType.int64} and self.quantization.mode != QuantizationMode.none:
            raise ValueError("quantization applies only to floating-point datasets")
        return self


class AlgorithmRequest(StrictModel):
    id: str
    params: dict[str, Any] = Field(default_factory=dict)


class CpuCoreClass(str, Enum):
    all = "all"
    performance = "performance"
    efficiency = "efficiency"


class CpuThreadPolicy(str, Enum):
    all_threads = "all_threads"
    one_thread_per_core = "one_thread_per_core"


class GpuControlMode(str, Enum):
    dedicated = "dedicated"
    shared = "shared"


class HardwareConfig(StrictModel):
    cpu_core_class: CpuCoreClass = CpuCoreClass.all
    cpu_thread_policy: CpuThreadPolicy = CpuThreadPolicy.one_thread_per_core
    cpu_numa_node: int | None = Field(default=None, ge=0)
    # Exact OS CPU IDs are an explicit, reproducible override for platforms where
    # automatic heterogeneous-core classification is unavailable or intentionally
    # not used. When present, this selection takes precedence over core/thread filters.
    cpu_explicit_ids: list[int] | None = None
    gpu_sets: list[Any] = Field(default_factory=lambda: ["each"])
    gpu_control_mode: GpuControlMode = GpuControlMode.dedicated
    memory_policy: Literal["default", "interleave"] = "interleave"

    @field_validator("cpu_explicit_ids")
    @classmethod
    def unique_cpu_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if any(v < 0 for v in value):
            raise ValueError("cpu_explicit_ids must contain non-negative OS CPU IDs")
        if len(value) != len(set(value)):
            raise ValueError("cpu_explicit_ids must not contain duplicates")
        if not value:
            raise ValueError("cpu_explicit_ids cannot be empty")
        return value


class MeasurementConfig(StrictModel):
    warmup_runs: int = Field(default=5, ge=1)
    # Repetitions per burst and independent bursts per calibration point for
    # profiled/adaptive schedulers. Each burst is reduced to a median, then the
    # median of burst-medians is used. Calibration remains outside TIMING.
    scheduler_calibration_repetitions: int = Field(default=5, ge=1, le=100)
    scheduler_calibration_bursts: int = Field(default=3, ge=1, le=9)
    # Fixed repetition count or an automatically sized timing batch. Auto mode runs
    # a short unrecorded probe after warm-up and targets timing_target_batch_seconds.
    timing_repetitions: int | Literal["auto"] = "auto"
    timing_probe_repetitions: int = Field(default=10, ge=3, le=1000)
    timing_target_batch_seconds: float = Field(default=0.5, gt=0.01, le=30.0)
    timing_min_repetitions: int = Field(default=30, ge=3)
    timing_max_repetitions: int = Field(default=100_000, ge=3, le=1_000_000)
    # Rotate between identical host replicas during warm-up/probe/timing/energy so
    # short CPU reductions do not benchmark one permanently hot cache-resident buffer.
    # Replicas are materialized before timing; pointer rotation itself is outside e2e timing.
    cache_rotation_target_bytes: int = Field(default=268_435_456, ge=0, le=4_294_967_296)
    cache_rotation_max_replicas: int = Field(default=64, ge=1, le=1024)
    energy_batch_repetitions: int | Literal["auto"] = "auto"
    energy_target_batch_seconds: float = Field(default=3.0, gt=0.1, le=120.0)
    energy_min_repetitions: int = Field(default=10, ge=1)
    energy_max_repetitions: int = Field(default=1_000_000, ge=1)
    blocks: int = Field(default=3, ge=1, le=100)
    randomization_seed: int = 20260813
    max_dataset_ram_fraction: float = Field(default=0.65, gt=0.05, le=0.95)
    # Preflight/runtime headroom for device allocations.  The estimator accounts for
    # input buffers; the remaining VRAM is intentionally left for CUB temporary storage,
    # CUDA context/runtime allocations and display/driver use.
    gpu_memory_safety_fraction: float = Field(default=0.80, gt=0.20, le=0.95)
    thermal_safety_gpu_c: float = Field(default=90.0, ge=40.0, le=110.0)
    thermal_safety_cpu_c: float = Field(default=95.0, ge=40.0, le=115.0)
    thermal_wait_timeout_s: float = Field(default=300.0, ge=0.0)
    worker_event_timeout_s: float = Field(default=1800.0, gt=0.0, le=86400.0)
    # Thesis-grade gate: convert environmental contamination warnings into hard failures.
    strict_preflight: bool = False
    max_preflight_cpu_load_percent: float = Field(default=5.0, ge=0.0, le=100.0)
    allow_gpu_graphics_processes: bool = False

    @model_validator(mode="after")
    def validate_repetitions(self) -> "MeasurementConfig":
        if isinstance(self.timing_repetitions, int):
            if not 3 <= self.timing_repetitions <= 1_000_000:
                raise ValueError("fixed timing_repetitions must be in [3, 1000000]")
        if self.timing_min_repetitions > self.timing_max_repetitions:
            raise ValueError("timing_min_repetitions cannot exceed timing_max_repetitions")
        if isinstance(self.energy_batch_repetitions, int):
            if not self.energy_min_repetitions <= self.energy_batch_repetitions <= self.energy_max_repetitions:
                raise ValueError(
                    "fixed energy_batch_repetitions must fall within "
                    "energy_min_repetitions/energy_max_repetitions"
                )
        if self.energy_min_repetitions > self.energy_max_repetitions:
            raise ValueError("energy_min_repetitions cannot exceed energy_max_repetitions")
        return self


class EnergyConfig(StrictModel):
    enable_cpu: bool = True
    enable_gpu: bool = True
    gpu_power_fallback_poll_ms: int = Field(default=100, ge=20, le=5000)


class TelemetryConfig(StrictModel):
    enabled: bool = True
    capture_cpu_frequency: bool = True
    capture_cpu_temperature: bool = True
    capture_gpu_state: bool = True
    # Snapshots are taken outside measured TIMING/ENERGY windows to avoid perturbing them.
    capture_pre_post_timing: bool = True
    capture_pre_post_energy: bool = True


class SystemBaselineConfig(StrictModel):
    capture_memory_inventory: bool = True
    # Optional external STREAM executable. Leave null when STREAM is not installed.
    # Standard Copy/Scale/Add/Triad lines are parsed into structured MB/s fields.
    stream_executable: str | None = None
    stream_args: list[str] = Field(default_factory=list)
    stream_timeout_s: float = Field(default=120.0, gt=1.0, le=1800.0)


class BuildConfig(StrictModel):
    build_dir: Path = Path("build")
    build_type: Literal["Release", "RelWithDebInfo"] = "Release"
    native_cpu_tuning: bool = True
    enable_cuda: Literal["auto", "on", "off"] = "auto"
    jobs: int | None = Field(default=None, ge=1)
    # Optional reproducible toolchain overrides. When CUDA is enabled the builder
    # deliberately uses one compiler for C++ and for NVCC host compilation.
    cxx_compiler: str | None = None
    cuda_host_compiler: str | None = None


class ExperimentGroup(StrictModel):
    id: str
    datasets: list[DatasetSpec]
    algorithms: list[AlgorithmRequest]
    operations: list[ReductionOperation] = Field(default_factory=lambda: [ReductionOperation.sum])
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @field_validator("operations")
    @classmethod
    def validate_operations(cls, value: list[ReductionOperation]) -> list[ReductionOperation]:
        if not value:
            raise ValueError("experiment operations cannot be empty")
        if len(value) != len(set(value)):
            raise ValueError("experiment operations must not contain duplicates")
        return value


class RootConfig(StrictModel):

    output_dir: Path = Path("results")
    dataset_cache_dir: Path = Path(".prbench/datasets")
    build: BuildConfig = Field(default_factory=BuildConfig)
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    system_baselines: SystemBaselineConfig = Field(default_factory=SystemBaselineConfig)
    experiments: list[ExperimentGroup]

    @field_validator("experiments")
    @classmethod
    def unique_experiment_ids(cls, value: list[ExperimentGroup]) -> list[ExperimentGroup]:
        ids = [x.id for x in value]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment group IDs must be unique")
        return value


class AlgorithmDefinition(StrictModel):
    id: str
    label: str
    scheduler: str
    cpu_backend: str | None = None
    gpu_backend: str | None = None
    transfer_policy: str | None = None
    requires_cuda: bool = False
    uses_cpu: bool
    uses_gpu: bool
    supports_multi_gpu: bool
    min_gpus: int = Field(default=0, ge=0)
    tunables: list[str] = Field(default_factory=list)
    role: str


class TopologyCpu(StrictModel):
    cpu_id: int
    socket_id: int
    core_id: int
    numa_node: int | None
    online: bool = True
    # Heterogeneous core classification. `unknown` is deliberate: the framework
    # never silently guesses P/E classes for a final research run.
    core_class: Literal["performance", "efficiency", "homogeneous", "unknown"] = "unknown"
    core_class_source: str = "unavailable"


class TopologyGpu(StrictModel):
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_bytes: int
    memory_free_bytes: int | None = None
    memory_used_bytes: int | None = None
    compute_capability: str | None = None
    numa_node: int | None = None
    local_cpus: list[int] = Field(default_factory=list)
    pcie_current_link_speed: str | None = None
    pcie_current_link_width: int | None = None
    pcie_max_link_speed: str | None = None
    pcie_max_link_width: int | None = None


class SystemTopologyModel(StrictModel):
    hostname: str
    os: str
    kernel: str
    machine: str
    logical_cpus: list[TopologyCpu]
    allowed_cpus: list[int]
    numa_nodes: dict[int, list[int]]
    gpus: list[TopologyGpu]
    total_ram_bytes: int
    nvml_available: bool
    p2p_matrix: dict[str, dict[str, Any]] = Field(default_factory=dict)
