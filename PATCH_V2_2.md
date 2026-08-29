# Parallel Reduction Benchmark v2.2 — zmiany po walidacji na serwerze hybrydowym

Wersja v2.2 odpowiada na wymagania ujawnione po poprawnym przejściu `smoke_cuda` i `smoke_all` na serwerze Intel P/E + NVIDIA RTX 4070 Ti oraz przygotowuje framework do serwerów wielo-GPU i wielogniazdowych.

## Najważniejsze zmiany

### 1. P/E-core jako jawny czynnik eksperymentalny

Konfiguracja sprzętu rozdziela teraz dwa niezależne wymiary:

```yaml
hardware:
  cpu_core_class: performance   # all | performance | efficiency
  cpu_thread_policy: all_threads # all_threads | one_thread_per_core
```

Klasa rdzenia nie jest inferowana z samej liczby siblingów SMT. Po zbudowaniu natywny worker przypina się kolejno do dozwolonych logicznych CPU i na wspieranych procesorach Intel odczytuje CPUID Hybrid Information Leaf `0x1A`. Na platformach, na których nie można wiarygodnie sklasyfikować rdzeni, `performance`/`efficiency` kończy plan błędem zamiast zgadywać. Dostępny jest jawny override `cpu_explicit_ids`.

### 2. Dedykowany lub współdzielony rdzeń sterujący GPU

Zastąpiono niejednoznaczne `reserve_gpu_control_cores` przez:

```yaml
gpu_control_mode: dedicated # albo shared
```

- `dedicated`: cały fizyczny rdzeń, włącznie z siblingami SMT, jest usuwany z puli redukcji CPU;
- `shared`: rdzeń pozostaje w puli redukcji CPU, więc koszt konkurencji z hostowym wątkiem GPU jest częścią badanego wariantu.

Każdy GPU otrzymuje osobny fizyczny rdzeń sterujący. Dwa GPU nie mogą zostać przypisane do dwóch logicznych siblingów tego samego rdzenia.

### 3. Topology-aware GPU↔CPU control assignment

Dla każdego GPU wybór rdzenia sterującego preferuje kolejno:

1. CPU z wybranej klasy rdzeni i z `local_cpus` urządzenia PCIe;
2. CPU z wybranej klasy rdzeni na lokalnym NUMA node;
3. pozostałe CPU z wybranej puli obliczeniowej;
4. dopiero potem kontrolowany fallback do innych dozwolonych CPU.

Dokładne powiązania są zapisywane w `gpu_control_bindings` wraz z GPU ID, CPU ID, socket/core/NUMA i źródłem lokalności.

### 4. Baseline'y multi-GPU

Dodano:

- `gpu_multi_cub_equal`;
- `gpu_multi_cub_profiled`.

Oba wymagają co najmniej 2 GPU i uruchamiają po jednym wątku hostowym oraz instancji CUB na każdym wybranym urządzeniu. Są potrzebne do uczciwego porównania `2×GPU` z `CPU+2×GPU`.

### 5. Dokładniejsze OpenMP affinity

Worker nie dziedziczy przypadkowej konfiguracji `OMP_*` użytkownika. Orchestrator wymusza `OMP_PLACES=threads` i `OMP_PROC_BIND=spread`, a proces jest ograniczony do dokładnej maski CPU zapisanej w wynikach.

### 6. `prbench plan`

Nowa komenda buduje worker, wykonuje natywną klasyfikację CPU i wypisuje cały plan bez pomiarów:

```bash
prbench plan --config configs/smoke_core_classes.yaml
```

Pozwala sprawdzić P/E, maski CPU, listy GPU oraz mapowanie GPU→CPU przed kosztowną kampanią.

### 7. `prbench preflight`

Przed finalnym badaniem można sprawdzić dostępność RAPL/NVML, klasy P/E, bieżące obciążenie CPU oraz obecność innych procesów compute na wybranych GPU:

```bash
prbench preflight --config <final-config.yaml>
```

Dla pomiarów energii obcy proces GPU jest błędem krytycznym. Dokumentacja przypomina też, że RAPL mierzy cały pakiet CPU, dlatego finalna kampania powinna korzystać z wyłącznego dostępu do noda.

### 8. Run nie kończy się już pozornym sukcesem

W v2.1 komunikat `Finished` mógł pojawić się również wtedy, gdy pojedyncze taski zostały zapisane jako `failed` lub `invalid`. V2.2 drukuje `Task status counts` i zwraca kod błędu, jeśli choć jedna instancja nie przeszła lub zwróciła niepoprawny wynik.

## Nowe konfiguracje testowe

- `configs/smoke_core_classes.yaml` — P-only, E-only i all;
- `configs/smoke_multi_gpu.yaml` — test 2+ GPU;
- `configs/pilot_tuning.yaml` — wyłącznie etap strojenia, bez energii.

`configs/research_core.yaml` pozostaje jedynie eksploracyjnym sweepem i nie powinien być używany jako finalna kampania, ponieważ miesza strojenie parametrów z pomiarem potwierdzającym.
