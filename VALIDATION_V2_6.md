# Validation report — v2.6

Data przygotowania: 2026-08-21.

## Wykonane testy w środowisku przygotowawczym

1. `python -m pytest -q` — **24/24 PASS**.
2. `python -m compileall -q prbench` — PASS.
3. CPU-only `cmake -S . -B build -DPRBENCH_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release` — PASS.
4. `cmake --build build` — PASS.
5. `ctest --test-dir build --output-on-failure` — **1/1 PASS** (`worker_self_test`).
6. `python -m prbench.cli run --config configs/smoke_cpu.yaml` — **3/3 task instances OK** end-to-end.
7. W trakcie implementacji wykonano także test konfiguracji `timing_repetitions: auto`; liczba iteracji została dobrana różnie dla szybkich/wolnych backendów zgodnie z docelowym czasem batcha.

## Sprawdzone artefakty runu CPU

Potwierdzono utworzenie i obecność nowych pól w:

- `tasks.jsonl`: sequence index, timestamps, prepare/calibration/partition metrics;
- `repetitions.jsonl`: element counts, CPU/GPU work fractions;
- `telemetry.jsonl`: phased telemetry + provenance częstotliwości;
- `datasets.json`: dataset manifest;
- `summary.csv` i `block_summary.csv`;
- `manifest.json`: timestamp_end/run status + native build info.

## CUDA

Środowisko przygotowawcze nie ma NVCC/NVIDIA GPU. Z tego powodu CUDA v2.6 **nie jest oznaczone jako runtime-zweryfikowane**. Na docelowym serwerze należy wykonać co najmniej:

```bash
prbench run --config configs/smoke_operations_cuda.yaml
```

oraz sprawdzić, czy `manifest.json -> build.native_build_info.gpu_libraries` zawiera wykryte wersje CUB/CCCL/CUDART, czy `telemetry.jsonl` ma stan NVML/PCIe oraz czy `gpu_metrics[].elements` sumują się do właściwej liczby elementów.

## Energia

Pełny CPU energy path wymaga czytelnego package-level Intel RAPL `energy_uj`. Do czasu nadania praw przez administratora należy prowadzić timing-only albo GPU-only NVML energy validation. V2.6 nie wykonuje ENERGY batcha, jeśli żaden żądany komponent danego tasku nie jest mierzalny/wybrany.
