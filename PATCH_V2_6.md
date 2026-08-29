# Parallel Reduction Benchmark v2.6

Wersja 2.6 jest przebudową warstwy pomiarowej wynikającą z audytu rzeczywistego runu z 2026-08-21. Celem nie było dodanie kolejnych algorytmów, lecz doprowadzenie protokołu benchmarku do stanu, w którym można wyjaśnić *dlaczego* dany wariant wygrał, wykryć drift/DVFS oraz odtworzyć dokładny przebieg eksperymentu.

## Najważniejsze zmiany

| Potrzeba badawcza | Implementacja v2.6 | Gdzie trafia wynik |
|---|---|---|
| timestamp start/end tasku | UTC timestamp + monotonic duration każdego tasku | `tasks.jsonl` |
| jawna kolejność | `sequence_index` dla każdej instancji | `tasks.jsonl`, `repetitions.jsonl`, `energy_batches.jsonl`, `telemetry.jsonl`, `plan` |
| timestamp końca runu | finalizacja manifestu także po Ctrl+C | `manifest.json` |
| temperatura CPU/GPU | snapshoty przed/po tasku, TIMING i ENERGY | `telemetry.jsonl` |
| GPU clocks/P-state | graphics/SM/memory clocks, P-state, power, utilization | `telemetry.jsonl` |
| CPU frequency | per-CPU frequency wraz ze źródłem/provenance | `telemetry.jsonl` |
| liczba elementów CPU/GPU | liczniki w natywnych backendach/schedulerach | `repetitions.jsonl`, `summary.csv` |
| udział pracy CPU/GPU | `cpu_work_fraction`, `gpu_work_fraction` | `repetitions.jsonl`, summaries |
| kalibracja schedulerów | próbki kalibracyjne oraz ich throughput | `tasks.jsonl -> prepare_metrics` |
| model throughput | model liniowy `T(n)=intercept+slope*n` oraz throughput adaptacyjny | `prepare_metrics`, `worker_throughput_elements_s` |
| energy coverage | jawne `cpu_package`, `gpu`, `cpu_package+gpu`, `none` | `energy_batches.jsonl`, `summary.csv` |
| per-block summary | osobna agregacja każdego randomizowanego bloku | `block_summary.csv` |
| automatyczny TIMING | po warm-up krótki PROBE i dobór liczby iteracji do czasu docelowego | `tasks.jsonl` + config |
| dataset manifest | specyfikacja, ścieżki, SHA-256, referencje, `sum_abs`, condition number | `datasets.json` |
| PCIe generation/width | current/max link speed i width ze sysfs + snapshot current GPU state | `manifest.json`, `telemetry.jsonl` |
| P2P matrix | best-effort NVML + surowe `nvidia-smi topo -p2p` | `manifest.json` |
| CUB/CCCL version | compile-time macros z natywnego workera | `manifest.json -> build.native_build_info` |
| RAM speed/channels / STREAM | DMI/lshw/EDAC best-effort + opcjonalny zewnętrzny STREAM | `manifest.json` |

## Automatyczny batch TIMING

Wcześniej stałe `timing_repetitions: 30` dawało bardzo krótkie okna dla małych datasetów. V2.6 pozwala ustawić:

```yaml
measurement:
  timing_repetitions: auto
  timing_probe_repetitions: 10
  timing_target_batch_seconds: 0.5
  timing_min_repetitions: 30
  timing_max_repetitions: 100000
```

Po warm-up worker wykonuje niewliczany do wyników `PROBE`. Orchestrator dobiera liczbę właściwych powtórzeń tak, aby cały TIMING miał docelową długość, z ograniczeniem min/max. Informacja o średnim czasie probe i faktycznej liczbie powtórzeń jest zapisywana w `tasks.jsonl`.

## Telemetria i perturbacja pomiaru

Snapshoty telemetrii są wykonywane poza oknami TIMING i ENERGY. Nie są wykonywane przy każdej mikrosekundowej iteracji. Dzięki temu temperatury, zegary i P-state można wykorzystać do kontroli driftu/DVFS bez niepotrzebnego zwiększania narzutu właściwego benchmarku.

CPU frequency jest raportowane wraz ze źródłem. Preferowane jest `cpuinfo_avg_freq`, później `cpuinfo_cur_freq`, a na końcu `scaling_cur_freq`. Ostatnia wartość nie jest przedstawiana jako laboratoryjny pomiar APERF/MPERF; provenance pozostaje w rekordzie. Jeżeli na serwerze dostępny jest `turbostat`/PMU, można w przyszłości dołączyć dodatkową kampanię walidacyjną bez zmiany głównych timingów.

## Load balancing

Backendy raportują liczbę elementów rzeczywiście obsłużonych przez CPU oraz każdy GPU. Dla każdego powtórzenia v2.6 zapisuje m.in.:

```text
cpu_elements
gpu_elements_total
gpu_metrics[].elements
cpu_work_fraction
gpu_work_fraction
worker_throughput_elements_s
```

Dzięki temu analiza schedulera nie musi zgadywać podziału na podstawie samych czasów.

## Kalibracja i modele

`ready` event zawiera `prepare_metrics`. Każdy punkt kalibracyjny jest mierzony wielokrotnie (`scheduler_calibration_repetitions`, domyślnie 5), a do modelu trafia mediana:

- próbki kalibracyjne CPU/GPU (`elements`, `elapsed_us`, `throughput_elements_s`),
- parametry dopasowanego modelu liniowego czasu,
- finalny statyczny podział zakresów,
- początkowe throughputy adaptacyjnego schedulera.

Wariant adaptive dodatkowo raportuje finalne EMA throughput po każdym właściwym powtórzeniu.

## Semantyka energii

V2.6 rozdziela „energię zmierzonych komponentów” od „energii całego systemu”. Każdy batch ma `energy_coverage`. Program nie wykonuje już bezsensownego GPU-energy batcha dla tasku CPU-only, jeśli CPU energy jest wyłączone.

Jeśli `enable_cpu: true`, RAPL package-level jest mierzone także przy GPU-only — jest to celowe, ponieważ host CPU zużywa energię przy obsłudze GPU. Jeśli `enable_gpu: true`, GPU energia jest dodawana tylko dla rzeczywiście używanych urządzeń.

## Floating-point SUM jako wynik jakości numerycznej

Audyt ujawnił, że naiwna sekwencyjna suma 100M dodatnich `float32` może zatrzymać się na `2^24`. To zachowanie jest właściwością arytmetyki i kolejności redukcji, a nie awarią infrastruktury.

Dlatego v2.6:

- zachowuje `is_correct`, absolute/relative error i tolerance dla każdej próbki,
- zapisuje task-level `numerical_mismatch_count`,
- nie klasyfikuje floating-point SUM jako `failed`/`invalid` wyłącznie z powodu przekroczenia tolerancji,
- nadal traktuje mismatch dla redukcji całkowitoliczbowych oraz float MIN/MAX jako błąd fatalny.

`accumulator_semantics: native_input_dtype` jest zapisywane jawnie. Jeśli w późniejszej wersji dodamy np. akumulator FP64 dla wejścia FP32, będzie to osobny wariant semantyczny i nie zostanie pomylony z obecną implementacją.

## Dataset manifest

`datasets.json` jest tworzony przed randomizowaną częścią eksperymentu i zawiera pełną specyfikację każdego datasetu, jego SHA-256, referencje SUM/MIN/MAX, `sum_abs` oraz condition number sumy, kiedy jest zdefiniowany. Generacja nadal odbywa się przed timingiem, dzięki czemu nie zmienia stanu termicznego tylko dla pierwszego algorytmu używającego danego pliku.

## Projekt eksperymentu

`preflight` generuje ostrzeżenia dla typowych przypadków confoundingu, np. gdy P-core i E-core są porównywane z różnymi operacjami albo `dedicated` i `shared` nie mają tej samej listy operacji. To ostrzeżenie nie zastępuje projektu eksperymentu, ale chroni przed błędem znalezionym w poprzednim runie.

## Nowe pliki wynikowe

Każdy run zawiera teraz co najmniej:

```text
manifest.json
config.snapshot.yaml
datasets.json
tasks.jsonl
repetitions.jsonl
telemetry.jsonl
energy_batches.jsonl        # jeśli wykonywano ENERGY
summary.csv
block_summary.csv
stderr/
```

## Ograniczenia

- Pełna ścieżka CUDA v2.6 musi zostać zweryfikowana na serwerze NVIDIA; środowisko przygotowawcze nie ma GPU/NVCC.
- NVML P2P API nie jest dostępne w każdej kombinacji biblioteki/sterownika. Wtedy manifest zachowuje `unknown` i surowy wynik `nvidia-smi topo`.
- DMI/`dmidecode` może wymagać uprawnień. Brak dostępu nie blokuje runu.
- STREAM nie jest wbudowywany ani automatycznie pobierany. Benchmark może uruchomić wskazany zewnętrzny, zweryfikowany executable i zapisać Copy/Scale/Add/Triad. To celowo nie miesza obcego kodu benchmarkowego z własnym workerem.
- Telemetria CPU frequency jest best-effort i raportuje źródło. Do osobnej walidacji częstotliwości można wykorzystać APERF/MPERF/turbostat, jeśli administrator udostępnia odpowiedni interfejs.
