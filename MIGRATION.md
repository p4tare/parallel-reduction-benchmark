# Migracja z wersji pierwotnej

## Główne zmiany architektoniczne

1. Usunięto kompilowanie osobnego wrappera dla każdego algorytmu. CMake buduje jeden wersjonowany `prbench-worker`.
2. Algorytm jest kompozycją trzech niezależnych strategii:
   - scheduler,
   - CPU backend,
   - GPU backend + transfer policy.
3. Python odpowiada za konfigurację, topologię, energię, randomizację i zapis wyników.
4. C++ odpowiada za właściwą ścieżkę obliczeniową i precyzyjny timing.
5. Stdout workera jest wyłącznie JSONL. Diagnostyka trafia na stderr.
6. Energia jest mierzona w batchach po warm-upie. Nie jest kopiowana jako „energia pojedynczej iteracji”.
7. Dane są deterministyczne, mają seed, hash i metadane referencyjne.
8. CPU affinity jest oparta o parę `(socket_id, core_id)`, a GPU locality o PCI sysfs/NUMA.
9. GPU bez CUDA/NVML nie powoduje cichego fallbacku na GPU0 — wariant jest pomijany albo zgłaszany jako niedostępny.
10. `float32_X` zastąpiono osobnymi polami `dtype` i `quantization`.

## Mapowanie stare -> nowe

- EXP001 -> `gpu_global_atomic`
- EXP002 -> `gpu_shared_naive`
- EXP012 -> `hybrid_static_equal`
- EXP013 manual 1:15 -> usunięty z core
- EXP013 v1 -> `hybrid_static_profiled`
- EXP013 v2 / EXP014 -> `hybrid_static_profiled_async`
- EXP013 v3 -> poza core
- EXP015 -> `hybrid_dynamic_fixed`
- EXP016 -> poza core
- EXP017 -> rozdzielony na `hybrid_dynamic_guided` i `hybrid_dynamic_adaptive`
- EXP018 -> poza core
