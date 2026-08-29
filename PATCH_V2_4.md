# v2.4 — operation-generic reductions, strict energy preflight and visible progress

## Why this release exists

A real server run exposed two issues that must be fixed before confirmatory measurements:

1. package-level Intel RAPL was present in sysfs but not readable by the benchmark user. `preflight` correctly reported this, but `run` could still be started and silently produced energy rows with CPU energy unavailable;
2. piping `prbench run ... | tee ...` changed Python stdout from a terminal to a pipe, so progress messages were block-buffered. A legitimate multi-second energy batch therefore looked like a hang.

The research scope was also expanded from sum-only reductions to multiple associative reduction operations.

## Changes

### Reduction operation is now a first-class experimental dimension

Experiment groups accept:

```yaml
operations: [sum, min, max]
```

The operation is propagated through the sweep planner, task identity, native worker CLI, CPU backend, GPU backend, hybrid schedulers, validation and result files. The same scheduling strategies can therefore be tested without duplicating algorithms.

The current built-in operations are:

- `sum`
- `min`
- `max`

Adding a future associative operation requires implementing its identity/combine semantics in the operation layer; schedulers and experiment orchestration do not need to be copied.

### CPU backends

`cpu_seq`, `cpu_omp` and `cpu_omp_simd` implement all three operations. OpenMP uses the matching `+`, `min` or `max` reduction clause.

### CUDA backends

`gpu_global_atomic`, `gpu_shared_naive`, `gpu_warp_atomic`, `gpu_two_pass` and `gpu_cub` are operation-generic. CUB dispatches to `DeviceReduce::Sum`, `Min` or `Max`. Min/max atomic baselines use CAS-based atomic combine for floating-point values, while sum retains the previous atomic-add path.

Hybrid and multi-GPU schedulers combine partial values with the selected operation identity rather than assuming zero/addition.

### Dataset references and validation

Dataset cache schema is bumped to v3 and metadata contains:

- `reference_sum`
- `reference_min`
- `reference_max`
- `sum_abs`

Min/max validation is exact for the finite generated inputs because no arithmetic reordering is involved. Sum retains the existing floating-point tolerance policy and raw error reporting.

### `run` now enforces preflight

A research run no longer starts when a requested energy source is unavailable. In particular, if `energy.enable_cpu: true` and no readable package-level RAPL counter exists, `prbench run` exits before dataset generation and measurements.

This prevents apparently successful runs that contain `cpu_energy_method: unavailable`.

### Better RAPL diagnostics

`prbench doctor` and `prbench preflight` now report candidate package RAPL paths, readability, permissions/uid/gid, samples or the concrete read error. This distinguishes "RAPL absent" from "RAPL exists but access is denied".

### Progress is line-buffered even through `tee`

All user-facing progress is flushed immediately. During energy measurement the runner prints the measured mean iteration time, chosen energy repetition count and estimated batch duration, then confirms completion of the batch.

### Energy watchdog

The ENERGY response uses a bounded timeout derived from the estimated batch duration instead of waiting the full generic worker timeout for a pathological batch.

### Clean Ctrl+C handling

Interrupted tasks are recorded with status `interrupted`, partial result files are retained, and the CLI exits with code 130 without a Python traceback.

## Validation performed in the build environment

- 16 Python tests pass.
- CPU-only Release build passes.
- native CTest/self-test passes for sum/min/max.
- CPU operation smoke passes 18/18 tasks across float32 and int32.

CUDA 12.4 execution of the new min/max paths must still be smoke-tested on the target NVIDIA server because the build environment used to prepare this package has no CUDA compiler/GPU.
