# Validation status — v2.3

## Local validation performed for this package

- `PYTHONPATH=. python -m pytest -q` → **13 passed**.
- CPU-only CMake Release build → passed.
- `ctest --test-dir <cpu-build> --output-on-failure` → native `worker_self_test` passed.
- Full CPU smoke path (`prbench run --config configs/smoke_cpu.yaml`) → **3/3 tasks ok**.
- Timing-only worker protocol verified: when both energy meters are disabled, the ENERGY batch is skipped, `energy_batch_repetitions=0`, and no energy JSONL is fabricated.
- Every YAML shipped in `configs/` was parsed by the strict configuration loader.
- `configs/pilot_tuning_full.yaml` plans 2136 task instances on the uploaded one-GPU/24-logical-CPU topology; this is intentionally large and exploratory.
- `configs/regression_dynamic_fixed.yaml` plans 12 task instances on that topology.

## Real-server evidence incorporated into v2.3

The user's v2.2 validation server successfully built the common C++/CUDA worker with CUDA 12.4 and GCC 13.4, classified its Intel hybrid CPU through native CPUID, and successfully executed all tested variants except one deterministic `hybrid_dynamic_fixed` parameter case.

The uploaded failed-run archive showed that every failure was `hybrid_dynamic_fixed` with `chunk_size=262144`; larger fixed chunks passed. The root cause was identified and fixed in v2.3. See `SERVER_RUN_ANALYSIS.md` and `PATCH_V2_3.md`.

## CUDA validation still required for v2.3

This package-preparation environment has no NVIDIA GPU/NVCC, so the changed v2.3 CUDA interaction must be smoke-tested on the target server. The first command should be:

```bash
prbench run --config configs/regression_dynamic_fixed.yaml
```

Expected on the current one-GPU server: **12/12 `ok`**. Then rerun `smoke_cuda`, `smoke_all`, and the complete tuning pilot from the same v2.3 source revision.

The multi-GPU paths remain to be validated on the second server with at least two visible NVIDIA GPUs.

## v2.6 — walidacja warstwy pomiarowej

W środowisku przygotowawczym wykonano:

- `pytest` — testy Python;
- CPU-only CMake Release build;
- natywny CTest/self-test;
- `configs/smoke_cpu.yaml` end-to-end;
- dodatkowy test `timing_repetitions: auto`, potwierdzający dobór różnej liczby iteracji zależnie od czasu algorytmu.

Ścieżka CUDA v2.6 wymaga ponownego `smoke_operations_cuda.yaml` na serwerze z NVIDIA GPU. W szczególności należy zweryfikować nowe liczniki `DeviceMetrics.elements`, compile-time build-info CUB/CCCL oraz snapshoty NVML/PCIe na rzeczywistym sterowniku.
