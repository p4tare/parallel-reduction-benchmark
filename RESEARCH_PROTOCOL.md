# Protokół finalnego badania

## 1. Walidacja każdej maszyny

Na każdej maszynie badawczej należy zainstalować tę samą rewizję źródeł, a następnie wykonać kolejno:

```bash
prbench doctor
prbench run --config configs/smoke_cpu.yaml
prbench run --config configs/smoke_cuda.yaml
prbench run --config configs/smoke_all.yaml
```

Na procesorze P/E dodatkowo:

```bash
prbench plan --config configs/smoke_core_classes.yaml
prbench run --config configs/smoke_core_classes.yaml
```

Na maszynie z co najmniej dwoma GPU dodatkowo:

```bash
prbench plan --config configs/smoke_multi_gpu.yaml
prbench run --config configs/smoke_multi_gpu.yaml
```

Run jest uznawany za poprawny wyłącznie wtedy, gdy `Task status counts` nie zawiera `failed` ani `invalid`.

## 2. Oddzielenie strojenia od wyników końcowych

Najpierw uruchamia się `configs/pilot_tuning.yaml`. Wyniki pilota służą tylko do wyboru parametrów (`block_size`, pipeline, chunk size, target chunk time, EMA). Nie wolno później traktować tych samych próbek jako finalnych wyników pracy.

Po wyborze parametrów należy je zamrozić w osobnych konfiguracjach finalnych. Zmiana parametrów po obejrzeniu wyników końcowych wymaga ponownego potraktowania kampanii jako pilotażu.

## 3. Zalecane rozdzielenie kampanii finalnej

Nie należy wykonywać jednego ogromnego iloczynu kartezjańskiego. Lepsze są trzy niezależne rodziny eksperymentów:

- **CPU**: `cpu_seq`, `cpu_omp`, `cpu_omp_simd`; P-only, E-only, all; bez GPU.
- **GPU/micro**: pięć backendów pojedynczego GPU na każdym GPU osobno oraz baseline'y multi-GPU na wszystkich GPU; CPU służy wyłącznie do host control.
- **Hybrid/macro**: sześć strategii CPU+GPU, wybrane GPU `[each, all]`, trzy klasy CPU tam, gdzie są dostępne, oraz osobne porównanie `gpu_control_mode=dedicated` i `shared`.

Takie rozdzielenie zapobiega wielokrotnemu powtarzaniu GPU-only algorytmów dla nieistotnych wariantów P/E.

## 4. Dane

Finalna kampania powinna obejmować co najmniej kilka rozmiarów w skali logarytmicznej, aby pokazać przejście od dominacji narzutu do ograniczenia przepustowością pamięci. Praktyczny zakres to np. `10^4`, `10^5`, `10^6`, `10^7`, `10^8`, z dodatkowym największym rozmiarem dopasowanym do RAM obu maszyn.

Rozmiar maksymalny musi być identyczny w porównaniach pomiędzy algorytmami na danej maszynie i nie powinien powodować swapowania. `max_dataset_ram_fraction` jest zabezpieczeniem, nie mechanizmem strojenia.

Dla każdej kombinacji należy zachować ten sam dataset seed i hash, aby wszystkie algorytmy redukowały identyczny wektor.

## 5. Powtórzenia i randomizacja

Dla finalnego pomiaru zalecane minimum:

```yaml
measurement:
  warmup_runs: 5
  timing_repetitions: 30
  blocks: 5
  energy_batch_repetitions: auto
  energy_target_batch_seconds: 3.0
```

Daje to 150 surowych próbek czasu na warunek, a jednocześnie pięć niezależnych batchy energetycznych. Kolejność jest randomizowana osobno w każdym bloku.

## 6. Warunki środowiskowe

Finalne pomiary energii należy wykonywać przy wyłącznym dostępie do CPU node i badanych GPU. RAPL obejmuje energię całego pakietu procesora, więc obciążenie innych użytkowników zanieczyszcza wynik nawet wtedy, gdy benchmark ma poprawne CPU affinity.

Przed każdym finalnym runem:

```bash
prbench preflight --config <config-final.yaml>
```

Należy zachować `manifest.json`, `config.snapshot.yaml`, wszystkie JSONL oraz stderr. Nie wolno łączyć wyników z różnych commitów źródła lub zmienionych konfiguracji bez potraktowania ich jako osobnych kampanii.

## 7. Dwie maszyny

Wyniki z dwóch serwerów nie powinny być wrzucane do jednego wspólnego rozkładu. Serwer jest osobnym czynnikiem eksperymentalnym. Najpierw raportuje się wyniki i ranking algorytmów na każdej maszynie, a następnie porównuje, które wnioski są powtarzalne między platformami.

Maszyna z 2 GPU powinna zawierać co najmniej trzy warianty GPU allocation dla algorytmów obsługujących wiele GPU: GPU0, GPU1 i `[GPU0,GPU1]`. Dzięki baseline'om `gpu_multi_cub_*` można wtedy rozdzielić efekt dodatkowego GPU od efektu dołączenia CPU.

## 8. Pełna przestrzeń parametrów a konfiguracja finalna

`configs/pilot_tuning_full.yaml` jest zachowaną „na zapas” szeroką przestrzenią kandydatów. Zawiera wszystkie dozwolone przez schemat rozmiary CUDA block (`32..1024`, potęgi 2) oraz szerokie siatki pipeline/chunk/guided/adaptive. Parametry ciągłe i dodatnie liczby całkowite mają nieskończenie wiele poprawnych wartości, dlatego ich dosłowne pełne wyliczenie nie jest możliwe; dokładne dziedziny opisuje `CONFIGURATION_REFERENCE.md`.

Finalne wyniki nie powinny pochodzić z tej pełnej siatki. Po pilocie wybrane wartości należy zamrozić w osobnych plikach CPU/GPU/hybrid i nie zmieniać ich po obejrzeniu kampanii potwierdzającej.

## v2.6 — dodatkowe wymagania protokołu

1. Finalne runy powinny używać `timing_repetitions: auto` z wcześniej zadeklarowanym docelowym czasem batcha. Faktyczna liczba iteracji jest zapisywana per task.
2. Nie interpretować `measured_component_energy_*` bez `energy_coverage`. `gpu` nie oznacza całkowitej energii algorytmu hybrydowego.
3. Przed analizą porównać `block_summary.csv` oraz `telemetry.jsonl` w funkcji `sequence_index`. Widoczny drift temperatury/zegarów/P-state należy opisać lub powtórzyć kampanię po stabilizacji środowiska.
4. Dla schedulerów hybrydowych raportować zarówno E2E, jak i `cpu_elements`, `gpu_elements_total` oraz work fractions. Bez tego nie należy interpretować różnic jako efektu load balancingu.
5. Dla profiled/adaptive archiwizować `prepare_metrics`; są częścią definicji algorytmu w danym runie.
6. Floating-point SUM analizować dwutorowo: wydajność + jakość numeryczna. Przekroczenie tolerancji jest wynikiem numerycznym, jeśli infrastruktura wykonała algorytm poprawnie.
7. Jeżeli badanym czynnikiem jest P/E/all albo dedicated/shared, pozostałe czynniki muszą być symetryczne. Ostrzeżenia preflight są pomocą, a nie substytutem projektu eksperymentu.
8. Dla porównań memory-bound zapisać albo wiarygodną inwentaryzację RAM, albo wynik STREAM (preferowane oba, jeśli polityka serwera na to pozwala).
