# Parallel Reduction Benchmark v2.6.1

Mała poprawka kompatybilności i planowania szerokich sweepów po pierwszej kompilacji CUDA v2.6 na CUDA 12.4 / GCC 13.4.

## Naprawione

1. **NVCC diagnostic #20013-D**
   - `reduction_identity()` nie wywołuje już `std::numeric_limits<T>::max()/lowest()` z funkcji `__host__ __device__`.
   - Zamiast włączać eksperymentalne `--expt-relaxed-constexpr`, używane są dokładne, compile-time stałe dla `int32`, `int64`, `float32` i `float64`.
   - Zachowuje to identyczną semantykę elementów neutralnych i usuwa ostrzeżenia na CUDA 12.4 z libstdc++ 13.

2. **`FULL_SWEEP_REFERENCE.yaml` nie abortuje na legalnym sweepie zawierającym pojedyncze nielegalne pary.**
   - Dla parametrów-list planner tworzy iloczyn kartezjański i odrzuca tylko kombinacje niespełniające zależności, np. `pipeline_chunks < pipeline_streams`.
   - Jawna konfiguracja scalarna `pipeline_streams: 8, pipeline_chunks: 4` nadal kończy się błędem fail-fast.
   - Jeżeli wszystkie kombinacje sweepu są nielegalne, planner również kończy się czytelnym błędem.

3. Dodano testy regresyjne dla obu przypadków.

## Ważne

`FULL_SWEEP_REFERENCE.yaml` jest konfiguracją eksploracyjną, a nie smoke testem. Na pojedynczym GPU może wygenerować kilkanaście tysięcy instancji zadań i działać przez wiele godzin. Przed uruchomieniem wykonaj `prbench plan --config configs/FULL_SWEEP_REFERENCE.yaml`.
