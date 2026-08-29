# Final runbook v3.0

## 0. Zasada

Nie zbieraj finalnych danych z niecommitowanego kodu. Wszystkie maszyny powinny używać
tej samej rewizji źródeł; tunables mogą być zamrożone per maszyna po niezależnym tuningu,
ale muszą znaleźć się w commicie/tagu oraz w config snapshot.

## 1. Testy po checkout

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e '.[dev]'

PYTHONPATH=. python -m pytest -q

cmake -S . -B build-cpu -DPRBENCH_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

## 2. Diagnostyka maszyny

Nie ustawiaj maskującego/reorderującego `CUDA_VISIBLE_DEVICES`.

```bash
prbench doctor > doctor_v3.json
prbench plan --config configs/FINAL_VALIDATION_v3.yaml > plan_validation_v3.json
prbench preflight --config configs/FINAL_VALIDATION_v3.yaml > preflight_validation_v3.json
```

## 3. Full-stack correctness

```bash
prbench run --config configs/FINAL_VALIDATION_v3.yaml | tee validation_v3.log
```

Warunek przejścia: zero `failed` i zero `invalid`.

Następnie obowiązkowo sprawdź kontrakt numeryczny dla dużych SUM:

```bash
prbench plan --config configs/FINAL_NUMERICAL_VALIDATION_v3.yaml > plan_numerical_v3.json
prbench preflight --config configs/FINAL_NUMERICAL_VALIDATION_v3.yaml > preflight_numerical_v3.json
prbench run --config configs/FINAL_NUMERICAL_VALIDATION_v3.yaml | tee numerical_validation_v3.log
```

Również tutaj warunek przejścia to zero `failed` i zero `invalid`. Jeśli poprawna
implementacja przekracza tolerancję, nie wolno jej arbitralnie poluzować po obejrzeniu
finalnych wyników; należy najpierw poprawić/uzasadnić kontrakt walidacji, ponownie wykonać
ten gate i dopiero potem przejść dalej.

## 4. Niezależny tuning

Tuning nie jest materiałem confirmatory.

```bash
prbench preflight --config configs/FINAL_TUNING_v3.yaml > preflight_tuning_v3.json
prbench run --config configs/FINAL_TUNING_v3.yaml | tee tuning_v3.log
```

Wybierz robust parameter set na podstawie całej treningowej siatki, nie pojedynczego
najlepszego punktu. Zamroź:
- politykę CPU SMT,
- block_size czterech własnych kerneli GPU,
- pipeline_streams + pipeline_chunk_elements,
- chunk_size fixed,
- min_chunk/guided_factor,
- min/max chunk + target_chunk_ms + ema_alpha adaptive.

Zmień tylko te pola w dwóch `FINAL_RESEARCH_*.yaml`, zacommituj i oznacz tagiem.

## 5. Code/config freeze

Przykład:

```bash
git status
git add configs/FINAL_RESEARCH_ALGORITHMS_v3.yaml configs/FINAL_RESEARCH_SCALING_v3.yaml
git commit -m "Freeze final benchmark parameters for <machine>"
git tag -a thesis-benchmark-v3-<machine> -m "Final thesis benchmark configuration for <machine>"
git push origin HEAD --tags
```

Strict preflight odmówi finalnego runu przy brudnym drzewie Git.

## 6. Finalny ranking algorytmów

```bash
prbench plan --config configs/FINAL_RESEARCH_ALGORITHMS_v3.yaml > plan_algorithms_v3.json
prbench preflight --config configs/FINAL_RESEARCH_ALGORITHMS_v3.yaml > preflight_algorithms_v3.json
prbench run --config configs/FINAL_RESEARCH_ALGORITHMS_v3.yaml | tee research_algorithms_v3.log
```

## 7. Skalowanie i duże dane

```bash
prbench plan --config configs/FINAL_RESEARCH_SCALING_v3.yaml > plan_scaling_v3.json
prbench preflight --config configs/FINAL_RESEARCH_SCALING_v3.yaml > preflight_scaling_v3.json
prbench run --config configs/FINAL_RESEARCH_SCALING_v3.yaml | tee research_scaling_v3.log
```

## 8. Kryteria akceptacji finalnego runu

- wszystkie taski `ok`;
- zero wyników `invalid`;
- komplet wszystkich bloków;
- `energy_complete_for_requested_components=true`;
- brak obcych procesów i incydentów thermal gate;
- brak brakujących dataset hash/config/manifest;
- telemetryka nie pokazuje trwałego throttlingu;
- dla profiled/adaptive zachowane `prepare_metrics`, partition i work fractions;
- nie łączyć wyników z różnych commitów/tagów w jeden rozkład.

Każda maszyna jest osobnym czynnikiem eksperymentalnym. Najpierw powstaje ranking per
maszyna, dopiero potem analizowana jest powtarzalność wniosków między platformami.
