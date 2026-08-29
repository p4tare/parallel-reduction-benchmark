# Patch v2.7.0 -- multi-GPU research hardening

This release hardens the existing multi-GPU implementation before using it for thesis-grade measurements on a new server/workstation.

## Changes

- `gpu_sets: [pairs]` / `[all_pairs]` dynamically expands all two-GPU combinations; together with `each` and `all` the same YAML works on 2- and 3-GPU machines.
- topology records NVML total/free/used VRAM for every GPU.
- preflight checks dataset RAM capacity before generation and exact GPU input-allocation capacity before execution, with configurable VRAM headroom (`gpu_memory_safety_fraction`, default 0.80).
- preflight reports GPU energy-counter support, graphics/compute processes, selected multi-GPU sets and capacity estimates.
- non-identity `CUDA_VISIBLE_DEVICES` fails fast because v2.7 still uses one common integer namespace for CUDA and NVML device IDs; unset the mask for research runs.
- multi-GPU/profiled calibration is pinned to each GPU's topology-local control CPU and previous thread affinity is restored after each calibration.
- pure multi-GPU orchestration/merge thread is pinned deterministically instead of being allowed to migrate.
- multi-NUMA GPU sets emit an explicit warning: the current dataset is one shared host allocation. `memory_policy=interleave` is reproducible but is not per-GPU NUMA-local placement.
- multiple package-level RAPL zones are reported explicitly; CPU package energy is their sum.

## Scientific scope

The implementation still does not use GPU P2P for the reduction itself: each GPU receives its host partition independently and final scalar partials are merged on the CPU. P2P topology is recorded for interpretation. This is intentional and should be described as a host-partitioned multi-GPU baseline rather than a P2P collective reduction.
