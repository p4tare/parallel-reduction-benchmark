# Architecture and extension guide

## Design goals

The framework is intentionally split into a Python experiment-control layer and one native C++/CUDA worker. Algorithms are compositions of three orthogonal decisions:

1. **device backend** — how a CPU or GPU reduces one contiguous range;
2. **scheduler** — which device receives which range and when;
3. **transfer policy** — how host/device movement is executed.

This prevents an experiment from accidentally changing several variables at once.

## SOLID mapping

- **Single Responsibility**: configuration, topology, dataset generation, build, energy metering, sweep planning, execution protocol, validation and result storage are separate modules. Native CPU/GPU reducers only reduce ranges; schedulers only coordinate work.
- **Open/Closed**: new schedulers implement `IReductionStrategy`; new accelerator implementations implement `IGpuReducer`. Existing orchestration does not need to be rewritten.
- **Liskov Substitution**: all reducer implementations return the same `PartialResult`/metrics contract and all schedulers implement the same `prepare`/`run_once` contract.
- **Interface Segregation**: the scheduler does not depend on NVML/RAPL or YAML, and energy meters do not know algorithm details. CUDA-specific behavior is hidden behind `IGpuReducer`.
- **Dependency Inversion**: high-level strategies depend on backend interfaces rather than concrete CUDA kernels. The Python runner depends on the worker protocol rather than individual algorithm binaries.


## Reduction-operation abstraction

The operation (`sum`, `min`, `max`) is orthogonal to backend, scheduler and transfer policy. Native code carries an explicit `ReductionOperation`; every partial value is initialized with the operation identity and merged through operation-specific combine semantics. CPU and CUDA backends implement the primitive reduction, while schedulers remain operation-agnostic.

To add another associative reduction operation:

1. extend the operation enum/parser and define its identity/combine semantics;
2. implement the primitive in CPU and GPU backends (including any CUB/custom dispatch);
3. add dataset reference and validation semantics;
4. add correctness/smoke tests.

No static/dynamic scheduler copy should be created solely for a new operation. This keeps the experimental factors separated and preserves the Open/Closed design.

## Adding a GPU reduction backend

1. Add an enum/parser value in `native/include/prbench/gpu_backend.hpp` / native parsing code.
2. Implement the backend behind `IGpuReducer` in the CUDA translation unit (or a future HIP translation unit).
3. Keep setup/allocation outside `reduce()` whenever that setup is intentionally excluded from the measurement boundary.
4. Populate H2D, kernel, D2H, overhead and total timing fields consistently.
5. Add an algorithm entry to the single canonical catalog `prbench/data/algorithm_catalog.yaml`.
6. Add smoke and correctness tests.

## Adding a scheduler

1. Add a scheduler enum/parser value.
2. Implement `IReductionStrategy` in `native/src/strategy.cpp` (or split into a new source file when the family grows).
3. Reuse existing CPU/GPU backends instead of embedding kernel logic in the scheduler.
4. Define only scheduler-specific tunables in the algorithm catalog.
5. Verify identical end-to-end measurement boundaries against existing schedulers.

## Adding another accelerator ecosystem

The orchestration layer intentionally does not contain CUDA kernel logic. A HIP/ROCm implementation can provide another native backend and topology/energy adapter while preserving datasets, sweep planning, protocol, validation and result schemas. Cross-vendor results should be reported as separate platform strata unless measurement domains are demonstrably equivalent.

## Measurement protocol

The worker performs:

1. setup and warm-up;
2. `READY`;
3. a `TIMING R` phase producing retained per-repetition metrics;
4. when energy measurement is enabled, a separate `ENERGY K` batch with no per-repetition serialization;
5. `MEASURE_DONE` immediately after that energy workload;
6. `DUMP`, which serializes the stored timing repetitions after the energy counters have been stopped (or immediately after timing when energy is disabled).

This separation is deliberate: timing requires raw repetitions, while energy requires a sufficiently long, clean measurement interval.
