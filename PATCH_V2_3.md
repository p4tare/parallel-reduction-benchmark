# Patch v2.3 — server-validation fixes

## Evidence from the uploaded server runs

The three failed campaigns were consistent and isolated to one parameter value:

- `smoke_cuda`: only `hybrid_dynamic_fixed`, `chunk_size=262144` failed;
- `smoke_all`: the same algorithm/parameter failed;
- `pilot_tuning`: exactly four failures = 2 datasets x 2 randomized blocks, all with `hybrid_dynamic_fixed`, `chunk_size=262144`;
- `chunk_size=1048576` and `4194304` succeeded in both pilot blocks.

Worker stderr was identical:

`GPU reducer received more elements than configured max_elements`

## Root cause

`DynamicHybridStrategy::prepare()` allocated each GPU reducer with capacity equal to the runtime fixed chunk (`262144`), but then unconditionally called the throughput-calibration routine used by the adaptive scheduler. That calibration used up to `max(min_chunk_size, 1<<20) = 1048576` elements. The fixed scheduler did not consume this throughput estimate, so the calibration was both unnecessary and larger than the allocated GPU buffer.

## Fix

- Fixed and guided dynamic schedulers no longer execute throughput calibration.
- Only `hybrid_dynamic_adaptive` calibrates CPU/GPU throughput.
- Adaptive calibration size is explicitly clamped to `dataset_count` and `max_chunk_size`.
- GPU reducer capacity is the maximum of runtime chunk capacity and calibration capacity.
- A CPU-only native self-test now covers the calibration policy, so the logical regression is testable without CUDA hardware.
- `configs/regression_dynamic_fixed.yaml` reproduces the previously failing small fixed chunks on a real CUDA machine.

## Additional measurement-quality changes

1. When both CPU and GPU energy measurement are disabled, the worker now skips the ENERGY batch completely. Parameter-tuning campaigns no longer perform unrecorded extra reductions that can change the thermal state of subsequent tasks.
2. Automatic energy-batch length is estimated from the just-completed timing batch (`mean_iteration_us`) rather than warm-up timing. This avoids one-time OpenMP/CUDA startup costs distorting the requested energy window.
3. Every unique dataset is materialized before the randomized measurement sequence, preventing first-use dataset generation from becoming an order-dependent thermal confounder.
4. A failed task is printed immediately with task ID, algorithm, dataset, GPU set, parameters and exception instead of only appearing as a final aggregate count.
5. The duplicate algorithm catalog was removed. The sole canonical catalog is `prbench/data/algorithm_catalog.yaml`.
6. Added `CONFIGURATION_REFERENCE.md` with every categorical setting and exact numeric domains.
7. Added `configs/pilot_tuning_full.yaml`, a deliberately large finite exploratory grid covering every legal CUDA block size and broad candidate grids for pipeline/dynamic parameters.

## Validation performed without NVIDIA hardware

- Python test suite: 11 tests passed.
- CPU-only CMake Release build: passed.
- native `worker_self_test`: passed through CTest.
- CPU smoke run through the full Python/native protocol: 3/3 tasks `ok`.
- timing-only protocol verified with `energy_batch_repetitions=0` and no energy JSONL.
- all shipped YAML files validate through the strict Pydantic/configuration loader.

CUDA v2.3 still requires the target NVIDIA server test. Run `configs/regression_dynamic_fixed.yaml` first; the expected result is 12/12 `ok` on the current one-GPU machine.
