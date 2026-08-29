# Patch v3.0.0 — final thesis benchmark hardening

## Cel

v3.0.0 jest kandydatem do zamrożenia programu używanego w finalnych eksperymentach
pracy magisterskiej. Zmiany wynikają z audytu realnego przebiegu v2.7 (810/810 tasków
poprawnych infrastrukturalnie), który ujawnił przede wszystkim metodologiczny efekt
wielokrotnego redukowania tego samego małego bufora po warm-up.

## Najważniejsze zmiany

### Cache-fair timing i kalibracja

Worker tworzy przed pomiarem identyczne repliki hostowego datasetu i rotuje aktywny
wskaźnik przed każdym warm-up, PROBE, TIMING, ENERGY oraz próbką kalibracji schedulerów.
Łączny docelowy working set określa `cache_rotation_target_bytes`, a liczbę replik
ogranicza `cache_rotation_max_replicas`.

Kopiowanie replik jest poza `e2e_us`; wewnątrz czasu redukcji pozostaje tylko właściwe
rozwiązanie problemu. Worker raportuje `dataset_replica_count` i
`dataset_resident_bytes`.

### Correctness jest warunkiem rankingu

Każde przekroczenie tolerancji walidatora oznacza task `invalid`, również dla
floating-point SUM. Tolerancja SUM nadal uwzględnia dtype, N i skalę/cancellation danych,
ale wynik poza kontraktem numerycznym nie może konkurować w rankingu wydajności.

### GPU-only async CUB

Dodano `gpu_cub_async`: CUB DeviceReduce z pinned host staging i wieloma CUDA streams.
Pozwala oddzielnie zbadać wpływ overlapu transferu na czystą ścieżkę GPU.

Async pipeline obsługuje nowy `pipeline_chunk_elements`. Stały rozmiar chunku ogranicza
pinned/device staging niezależnie od całkowitego N. Stare `pipeline_chunks` pozostaje
wspierane dla zgodności ze starszymi konfiguracjami.

### Strict research gate

`measurement.strict_preflight: true` powoduje m.in.:
- wymaganie czystego drzewa Git,
- twardy limit wstępnego obciążenia CPU,
- brak obcych GPU compute processes,
- opcjonalny zakaz graphics processes,
- wymaganie użytecznej telemetrii energii GPU (counter lub power sampling),
- ponowne sprawdzenie CPU/GPU idleness przed każdym taskiem w wielogodzinnej kampanii.

### Finalne konfiguracje

- `FINAL_VALIDATION_v3.yaml` — correctness/full-stack gate, nie dane do pracy;
- `FINAL_TUNING_v3.yaml` — niezależny pilot do wyboru parametrów na maszynie;
- `FINAL_RESEARCH_ALGORITHMS_v3.yaml` — pełny ranking CPU + jeden GPU;
- `FINAL_RESEARCH_SCALING_v3.yaml` — duże N i skalowanie 1/2/3 GPU.

Finalne pliki research używają innych seedów niż tuning. Po tuningu wolno zmienić wyłącznie
jawne tunables i wybraną politykę SMT, następnie commit/tag; po rozpoczęciu confirmatory
runu parametry są zamrożone.

## NUMA

v3 rozróżnia politykę pamięci według scenariusza:
- CPU: interleave przy pracy na wielu NUMA nodes,
- single GPU: default first-touch po przypięciu host control thread lokalnie do GPU,
- multi-GPU: interleave jako kontrolowany wspólny host allocation,
- hybrid: interleave dla współdzielonego host datasetu.

Topologia pozostaje zapisana w manifeście; multi-GPU nadal nie wykonuje P2P reduction.
