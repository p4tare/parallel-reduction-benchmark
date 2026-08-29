# Reprodukcja eksperymentów

## 1. Środowisko

Docelowa platforma badawcza to Linux. CPU-only działa bez CUDA. Algorytmy GPU wymagają NVIDIA CUDA Toolkit i widocznego GPU CUDA. Pomiar energii CPU korzysta z Linux powercap/RAPL, a GPU z NVML.

## 2. Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## 3. Diagnostyka

```bash
prbench doctor
prbench list-algorithms
```

## 4. Test CPU

```bash
prbench run --config configs/smoke_cpu.yaml
```

## 5. Test pełnego katalogu

```bash
prbench run --config configs/smoke_all.yaml
```

Na serwerze CPU-only automatycznie pozostaną warianty CPU. Na serwerze CUDA zostaną również zaplanowane warianty GPU i hybrydowe. Dodatkowo można uruchomić konfigurację stricte CUDA:

```bash
prbench run --config configs/smoke_cuda.yaml
```

## 6. Walidacja topologii i strojenie

Na procesorze hybrydowym P/E:

```bash
prbench plan --config configs/smoke_core_classes.yaml
prbench run --config configs/smoke_core_classes.yaml
```

Na serwerze 2+ GPU:

```bash
prbench plan --config configs/smoke_multi_gpu.yaml
prbench run --config configs/smoke_multi_gpu.yaml
```

Następnie wykonaj osobny pilot strojenia parametrów:

```bash
prbench run --config configs/pilot_tuning.yaml
```

Próbek pilota nie należy używać jako wyników końcowych. Po zamrożeniu parametrów przygotuj osobne konfiguracje finalne CPU/GPU/hybrid i przed każdym runem wykonaj:

```bash
prbench preflight --config <final.yaml>
prbench plan --config <final.yaml> > plan.json
prbench run --config <final.yaml>
```

`configs/research_core.yaml` pozostaje eksploracyjnym sweepem i nie jest finalnym protokołem pracy. Szczegóły: `RESEARCH_PROTOCOL.md`.

## 7. Artefakty runu

Każdy katalog `results/run_*` zawiera:
- `manifest.json` — sprzęt, topologia, toolchain, commit, config;
- `config.snapshot.yaml` — niezmieniona konfiguracja wejściowa;
- `tasks.jsonl` — status każdej instancji;
- `repetitions.jsonl` — surowe czasy i wyniki;
- `energy_batches.jsonl` — energia całych okien pomiarowych;
- `summary.csv` — statystyki pochodne;
- `stderr/*.log` — logi diagnostyczne każdego taska.

## 8. Ważne zasady interpretacji

- `e2e_us` oznacza dane wejściowe już rezydujące w RAM i końcowy wynik w RAM. Nie obejmuje odczytu pliku, kompilacji, alokacji, warm-upu i kalibracji; koszty konstrukcji/przygotowania strategii są jednak zachowane osobno jako `strategy_create_us` i `prepare_us`.
- `gpu_kernel_sum_us` i czasy per-device są diagnostyczne; dla współbieżnych GPU nie wolno sumy czasu kerneli utożsamiać z czasem ściennym.
- energia jest wynikiem batcha i jest dzielona przez liczbę redukcji wyłącznie jako estymata `energy_per_reduction`.
- CUB jest baseline'em bibliotecznym; nie należy przedstawiać go jako własnego algorytmu.
- `hybrid_static_profiled_async` różni się od `hybrid_static_profiled` polityką transferu, dzięki czemu można izolować efekt overlapu.
