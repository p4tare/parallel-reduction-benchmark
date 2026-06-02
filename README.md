# System Redukcji Hybrydowej (CPU + GPU) ??

Projekt badawczy maj¹cy na celu rzetelne porównanie wydajnoœci i zu¿ycia energii (Performance & Energy Profiling) dla ró¿nych algorytmów redukcji danych (np. sumowanie, znajdowanie maksimum) w œrodowiskach hybrydowych.

## ?? O projekcie
System sk³ada siê z dwóch warstw:
1. **Orkiestrator (Python):** Wykrywa topologiê sprzêtow¹ (NUMA, P/E Cores, GPU), kompiluje kod JIT (Just-In-Time), zapobiega *thermal throttlingowi* i precyzyjnie profiluje zu¿ycie energii (Intel RAPL, Nvidia NVML).
2. **Warstwa Obliczeniowa (C++ / CUDA / OpenMP):** Wysoce zoptymalizowane kernele operuj¹ce w izolowanym œrodowisku przydzielonym przez orkiestratora, mierz¹ce czas wykonania z rozdzielczoœci¹ nanosekundow¹.

## ??? Wymagania
* **OS:** Linux (wymagany do narzêdzi RAPL, `perf` i `taskset`)
* **Python:** 3.10+
* **Kompilatory:** GCC/G++, CMake, Nvidia CUDA Toolkit (nvcc) >= 11.8

## ?? Struktura Projektu
* `configs/` - Pliki konfiguracyjne YAML dla poszczególnych przebiegów badawczych.
* `src/` - Kod Ÿród³owy Orkiestratora w Pythonie.
* `algorithms/` - Kody Ÿród³owe badanych algorytmów w C++/CUDA.
* `results/` - [Ignorowane w repozytorium] Miejsce zrzutu wyników (CSV/JSON).

## ?? Szybki start (Wkrótce)
Uruchomienie g³ównej puli eksperymentów:
```bash
python main.py --config configs/main_experiments.yaml
```