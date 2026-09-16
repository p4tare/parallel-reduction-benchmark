# Architecture and extension guide

## Design goals

Version 4 keeps one experiment-control stack and one native worker. There are no separate
"memory-path", "topology-stream" or "out-of-core" benchmark applications. Every research
variant is selected from the canonical algorithm catalog and is executed through:

`prbench run <config.yaml>` -> `ExperimentRunner` -> `prbench-worker`.

Algorithms are compositions of orthogonal decisions:

1. **device backend** — how a CPU or GPU reduces one contiguous range;
2. **scheduler/topology** — which CPU/GPU engines receive which ranges and when;
3. **transfer policy** — synchronous copy, asynchronous pipeline or device-resident use;
4. **memory path** — explicit copy, chunked copy, registered pinned input, zero-copy,
   Unified Memory policy, HMM, or GDS;
5. **storage policy** — host-resident input, bounded file streaming, or GPUDirect Storage.

Keeping these dimensions explicit prevents a study from silently changing several causal
factors at once and lets old and new algorithms use identical measurement infrastructure.

## One orchestration and result pipeline

All algorithms, including the v4 memory-path and out-of-core variants, use the existing
worker protocol and results pipeline. The Python layer is responsible for topology discovery,
preflight, dataset generation/cache, build provenance, task randomization, thermal/idleness
gates, telemetry, energy measurement, numerical validation and result serialization.

The native worker owns algorithm execution only. It exposes the same protocol for every
strategy:

1. native setup, calibration and warm-up;
2. `READY`;
3. optional `PROBE R` for automatic timing-batch sizing;
4. `TIMING R`;
5. optional `ENERGY K`;
6. `DUMP` retained timing repetitions;
7. `DONE`.

Consequently, new methods receive the same console progress, task counters, timing batch
selection, energy accounting, telemetry snapshots and JSONL/CSV result semantics as the
legacy v3 algorithms.

## Strategy families

Legacy host-resident algorithms continue to use `native/src/strategy.cpp` and the established
CPU/GPU reducer abstractions. Memory/storage paths that require ownership of special host or
managed allocations, bounded file I/O, or topology-aware staging implement the same
`IReductionStrategy` contract in `native/src/integrated_strategy.cu`.

This is an implementation split, not a second benchmark framework. Both strategy families
are created by the same `prbench-worker`, receive the same `WorkerConfig`, and emit the same
`IterationMetrics` schema.

The integrated family currently covers:

- registered host input (`cudaHostRegister`);
- synchronous bounded chunking;
- zero-copy mapped host memory;
- Unified Memory demand migration;
- Unified Memory explicit prefetch;
- Unified Memory advice policies;
- HMM/system-pageable access with runtime capability gating;
- CPU file streaming;
- single-GPU file streaming;
- concurrent multi-GPU file streaming;
- CPU + one/multiple GPU file-stream hybrids;
- NUMA-local registered staging when libnuma is available.

The established GPU reducer path continues to cover the ordinary explicit copy,
asynchronous pinned pipeline, device-resident CUB path and legacy custom CUDA kernels.

## Storage semantics

`storage_policy` is part of every algorithm definition and every result row.

- `host_resident`: the logical input is resident in system memory before the measured
  reduction interval. The configured RAM safety limit applies.
- `file_stream`: the dataset file is processed through bounded host/device buffers. The
  logical dataset may exceed RAM; storage capacity and filesystem behavior become part of
  the experiment.
- `gds`: storage is transferred directly to device memory through cuFile when the platform
  provides a valid GPUDirect Storage stack.

Host-resident and file-stream/GDS observations answer different research questions and must
not be pooled into one ranking without explicitly modelling the storage term.

## Memory-path semantics

`memory_path` is recorded in the algorithm catalog, worker command line and result metadata.
Platform-dependent mechanisms are capability-gated rather than silently replaced:

- HMM requires CUDA pageable system-memory access;
- zero-copy requires mapped host-memory support;
- Unified Memory variants require managed-memory support;
- NUMA-local staging requires libnuma and a known GPU NUMA node;
- GDS requires cuFile plus a compatible storage/kernel stack.

An unavailable optional mechanism is reported as unsupported/skipped. The benchmark must not
pretend that a fallback implementation measured the requested mechanism.

## SOLID mapping

- **Single Responsibility**: configuration, topology, dataset generation, build, energy
  metering, sweep planning, execution protocol, validation and result storage are separate
  modules. Native reducers/strategies do not own experiment orchestration.
- **Open/Closed**: new schedulers and memory/storage strategies implement
  `IReductionStrategy`; ordinary accelerator reducers implement `IGpuReducer`.
- **Liskov Substitution**: all strategies return the same `IterationMetrics` contract and all
  reducers return the same `PartialResult`/device-metrics contract.
- **Interface Segregation**: native strategies do not depend on YAML, NVML or RAPL; energy
  meters do not know algorithm implementation details.
- **Dependency Inversion**: the Python runner depends on the worker protocol and canonical
  algorithm metadata instead of algorithm-specific executables.

## Reduction-operation abstraction

The operation (`sum`, `min`, `max`) is orthogonal to backend, scheduler, memory path and
storage policy. Native code carries an explicit `ReductionOperation`; every partial value is
initialized with the operation identity and merged through operation-specific combine
semantics.

To add another associative reduction operation:

1. extend the operation enum/parser and identity/combine semantics;
2. implement the primitive in CPU and GPU backends (including CUB/custom dispatch);
3. add dataset reference and validation semantics;
4. add correctness/smoke tests.

No scheduler or orchestration copy should be created solely for a new operation.

## Adding a GPU reduction backend

1. Add an enum/parser value in the native type/parser layer.
2. Implement the backend behind `IGpuReducer`.
3. Keep intentionally excluded setup/allocation outside `reduce()`.
4. Populate H2D, kernel, D2H, overhead and total timing fields consistently.
5. Add an algorithm entry to `prbench/data/algorithm_catalog.yaml`.
6. Add smoke and correctness tests.

## Adding a memory or storage path

1. Add the path metadata to the canonical algorithm catalog.
2. Reuse an existing scheduler/reducer when the path is only a transport-policy change.
3. If the path owns special allocations or file I/O, implement an `IReductionStrategy` in the
   integrated strategy family rather than adding another executable/orchestrator.
4. Populate byte counters and component timings in `DeviceMetrics`/`CpuMetrics`.
5. Add explicit runtime capability gating for optional platform features.
6. Route the algorithm through the normal `ExperimentRunner` and worker protocol.
7. Add planner/capacity/config tests and an integrated smoke configuration.

## Adding a scheduler

1. Add a scheduler enum/parser value.
2. Implement `IReductionStrategy` in the native strategy layer.
3. Reuse existing CPU/GPU backends instead of embedding unrelated kernel logic.
4. Define only scheduler-specific tunables in the algorithm catalog.
5. Verify identical measurement boundaries against existing schedulers.

## Adding another accelerator ecosystem

The orchestration layer contains no CUDA kernel logic. A HIP/ROCm implementation can provide
another native backend and topology/energy adapter while preserving datasets, sweep planning,
protocol, validation and result schemas. Cross-vendor results should be reported as separate
platform strata unless measurement domains are demonstrably equivalent.

## Measurement protocol

Timing and energy are intentionally separate measurement windows. Timing retains raw
repetitions; energy uses a sufficiently long batch with no per-repetition serialization in the
measurement window. `DUMP` occurs after the energy counters have been stopped. This protocol
applies equally to legacy algorithms, memory-path variants, multi-GPU topologies and
out-of-core strategies.
