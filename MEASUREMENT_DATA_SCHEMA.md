# Dane zapisywane przez Parallel Reduction Benchmark v2.6

## `manifest.json`

Metadane całego runu: UTC start/end, status i duration, Git/source hash, hash binarki, Python/packages, OS/kernel, CPU/GPU/NUMA/PCIe topology, P2P matrix, toolchain, CUB/CCCL/CUDART, CPU governor/turbo, `lscpu`, `nvidia-smi`, pamięć RAM (best effort), opcjonalny STREAM i pełna konfiguracja znormalizowana przez Pydantic.

## `config.snapshot.yaml`

Bajtowa kopia YAML użytego do uruchomienia. Należy archiwizować razem z runem.

## `datasets.json`

Dla każdego unikalnego datasetu: oryginalna specyfikacja, seed, dtype, distribution, quantization, ścieżka cache, SHA-256, rozmiar, referencje SUM/MIN/MAX, `sum_abs`, condition number sumy (jeżeli zdefiniowany).

## `tasks.jsonl`

Jeden rekord dla każdej instancji warunku w każdym randomizowanym bloku. Najważniejsze pola:

- `sequence_index`, `block_index`, `task_key`, `task_instance_id`,
- algorytm, parametry, operacja, dataset,
- CPU affinity/core class/thread policy/NUMA,
- GPU IDs i control-core bindings,
- UTC start/end + duration,
- warm-up, strategy construction, prepare/calibration,
- `prepare_metrics`: kalibracja, model throughput, finalny statyczny partition,
- probe TIMING, rzeczywista liczba timing/energy repetitions,
- status oraz liczba mismatchów numerycznych.

## `repetitions.jsonl`

Surowe wyniki każdego powtórzenia TIMING. Oprócz E2E zawiera CPU compute, H2D/kernel/D2H per GPU, scheduler/merge, liczby chunków, liczby elementów CPU/GPU, work fractions, wynik/referencję i błędy numeryczne. Dla adaptive znajduje się również bieżący throughput workerów.

To jest podstawowe źródło do statystyki; `summary.csv` jest jedynie agregatem.

## `telemetry.jsonl`

Snapshoty poza właściwym oknem pomiarowym. Każdy rekord ma phase + UTC timestamp + task metadata. Zależnie od konfiguracji zawiera:

- per-CPU MHz i provenance,
- CPU temperatures,
- GPU temperature/P-state/clocks/power/utilization/current PCIe link.

Pozwala sprawdzać drift termiczny i DVFS w kolejności `sequence_index`.

## `energy_batches.jsonl`

Jeden rekord dla osobnego batcha energetycznego. Zawiera liczbę redukcji, czas batcha, CPU package energy, GPU energy per urządzenie, metodę pomiaru, `energy_coverage` oraz energię/redukcję tylko dla rzeczywiście zmierzonych komponentów.

## `summary.csv`

Agregacja per warunek: median/mean/P25/P75/min/max dla timingów, elementów, work fraction i błędów numerycznych; liczba prób, correct fraction, mediana energii i EDP z jawnym `energy_coverage`.

## `block_summary.csv`

Te same główne agregaty osobno dla każdego randomizowanego bloku. Umożliwia wykrycie trendu w czasie i różnic termicznych/order effects bez niszczenia raw data.

## `stderr/`

Log per task instance. Zachowywać nawet dla udanego runu — ostrzeżenia CUDA/driver/runtime mogą być użyteczne przy późniejszym audycie.
