# Dobór algorytmów do badania

## Stan wyjściowy repozytorium

Analiza dotyczy gałęzi `main`, commit `22ae0b5acafc7f833793fb5f4b8eac822bb3a83f`.
W bieżącym drzewie znajdują się: EXP001, EXP002, EXP012, EXP013 (wariant podstawowy i v1-v3), EXP014 oraz EXP015-EXP018. Nie ma kompletnego zestawu „20 algorytmów” ani pełnej grupy CPU-only.

## Ocena obecnych wariantów

### EXP001 — Global Memory Atomics
**Decyzja: zachować jako celowo naiwny baseline.**

Zalety:
- bardzo prosta semantyka;
- dobrze pokazuje wpływ kontencji na jednym globalnym akumulatorze;
- wartościowy punkt dydaktyczny i dolna granica jakości implementacji GPU.

Wady:
- nie jest konkurencyjną implementacją produkcyjną;
- przy dużej liczbie wątków atomiki na jednym adresie dominują wykonanie;
- obecna wersja ma własny `main.cu` i jest niezgodna z aktualnym wrapperem.

### EXP002 — Naive Shared Memory
**Decyzja: zachować jako baseline pamięci współdzielonej.**

Zalety:
- izoluje korzyść redukcji liczby atomików względem EXP001;
- reprezentuje klasyczne drzewo w pamięci współdzielonej.

Wady:
- interleaved addressing i liczne bariery są celowo nieoptymalne;
- obecna implementacja wykorzystuje tylko pierwszy GPU worker;
- opis pracy sugeruje „paczki danych na wątek”, czego obecny kod nie realizuje.

### EXP012 — Static Equal
**Decyzja: zachować koncepcję, przepisać implementację.**

Zalety:
- konieczny baseline dla strategii hybrydowych;
- brak fazy uczenia/profilowania;
- łatwa interpretacja.

Wady obecnej implementacji:
- CPU nie jest mocnym, kontrolowanym backendem porównawczym;
- stary interfejs `execute_algorithm`;
- równa liczba elementów nie odpowiada różnej przepustowości CPU/GPU.

### EXP013 — stała waga CPU:GPU = 1:15
**Decyzja: usunąć z badania głównego.**

Arbitralna stała waga jest zależna od konkretnego serwera i osłabia generalizowalność wyniku. Jeżeli ma pozostać, wyłącznie jako eksperyment pomocniczy `static_manual_weight`, nie jako „profilometryczny”.

### EXP013 v1 — Profilometric Load Balancing
**Decyzja: zachować ideę, zastąpić modelem kosztu.**

Zalety:
- uzasadniony kierunek: podział na podstawie pomiaru urządzeń;
- znacznie lepszy od stałej 1:15.

Wady obecnej wersji:
- próbka 100k może być zdominowana przez narzuty stałe GPU;
- pilot nie odwzorowuje dokładnie ścieżki właściwego benchmarku;
- zagnieżdżony OpenMP utrudnia kontrolę zasobów.

W v2 zastosowano kalibrację dla kilku rozmiarów oraz model `T(N)=a+bN`, a podział jest wyznaczany na podstawie przewidywanego makespanu.

### EXP013 v2 — Profilometric + Overlap
**Decyzja: nie traktować jako nowego schedulera; zachować jako politykę transferu.**

Overlap jest osobnym wymiarem eksperymentalnym. Łączenie go z nowym algorytmem podziału utrudnia wskazanie źródła przyspieszenia. W v2 jest to wariant `hybrid_static_profiled_async`: ten sam podział co `hybrid_static_profiled`, ale inna polityka transferu.

### EXP013 v3 — Hierarchical P2P
**Decyzja: usunąć z badania głównego.**

Powody:
- zależy od topologii multi-GPU i dostępności P2P;
- miesza trzy czynniki: podział pracy, overlap oraz agregację P2P;
- bieżąca wersja ma błędy w ścieżce P2P i agregacji;
- dla pracy o CPU+GPU nie jest potrzebny do odpowiedzi na główne pytania badawcze.

Może wrócić jako rozszerzenie po ustabilizowaniu pomiarów.

### EXP014 — Async Pipelining
**Decyzja: zachować ideę jako kontrolowany wariant transferu.**

Zalety:
- pozwala zbadać overlap H2D/kernel/D2H;
- ważny dla redukcji o niskiej intensywności arytmetycznej.

Wady obecnej wersji:
- używa ręcznej wagi 1:15;
- jest traktowany jako osobny „algorytm”, mimo że główna różnica to polityka transferu;
- wynik zależy od poprawnego użycia page-locked memory.

### EXP015 — Centralized Task Queue
**Decyzja: zachować jako dynamiczny baseline.**

Zalety:
- prosta interpretacja;
- naturalne samobalansowanie wynikające z częstszego pobierania zadań przez szybsze urządzenia;
- dobry punkt odniesienia dla bardziej zaawansowanych schedulerów.

Wady obecnej wersji:
- arbitralne `N/100` jako chunk size;
- synchroniczna ścieżka GPU per chunk;
- CPU nie jest odseparowany od host-workerów GPU.

### EXP016 — Distributed Work Stealing
**Decyzja: przenieść poza badanie główne.**

Dla jednego „silnika CPU” i kilku GPU centralna kolejka już daje naturalne dynamiczne rozdzielanie pracy. Work stealing dodaje złożoność i kolejne atomiki bez jasnej hipotezy, którą samodzielnie testuje. Warto wrócić do niego dopiero, jeśli wyniki centralnej kolejki pokażą realny bottleneck scheduler'a.

### EXP017 — Adaptive Chunk Sizing
**Decyzja: zachować po zmianie nazwy i dodać prawdziwy wariant adaptacyjny.**

Bieżący algorytm jest bliższy guided self-scheduling: chunk zależy od liczby pozostałych elementów, nie od zmierzonej szybkości workerów. W v2 rozdzielono:
- `hybrid_dynamic_guided` — guided self-scheduling;
- `hybrid_dynamic_adaptive` — rozmiar kolejnej porcji zależy od EMA zmierzonej przepustowości danego workera i docelowego czasu porcji.

### EXP018 — Cross-Socket P2P
**Decyzja: usunąć z badania głównego.**

Jest silnie zależny od topologii konkretnego serwera, a brak P2P zmienia semantykę i koszt ścieżki wykonania. Obecna implementacja nie weryfikuje poprawnie możliwości peer access przed użyciem zdalnych wskaźników. Jest to interesujący eksperyment dodatkowy, ale nie powinien wpływać na główny ranking strategii CPU+GPU.

---

# Zalecany zestaw główny

## Mikro — CPU
1. `cpu_seq` — sekwencyjny punkt odniesienia.
2. `cpu_omp` — wielowątkowa redukcja OpenMP.
3. `cpu_omp_simd` — mocny baseline CPU z wielowątkowością i SIMD.

## Mikro — GPU
4. `gpu_global_atomic` — celowo naiwny atomik per wątek.
5. `gpu_shared_naive` — naiwny shared-memory tree.
6. `gpu_warp_atomic` — rejestry + warp shuffle + redukcja blokowa + jeden atomik na blok.
7. `gpu_two_pass` — hierarchiczna redukcja wieloprzebiegowa bez globalnej kontencji atomików pomiędzy blokami.
8. `gpu_cub` — CUB DeviceReduce jako nowoczesny baseline biblioteczny.

## Makro — CPU+GPU
9. `hybrid_static_equal` — równy statyczny podział.
10. `hybrid_static_profiled` — podział na podstawie modelu `T(N)=a+bN` dla każdego urządzenia.
11. `hybrid_static_profiled_async` — ten sam podział, ale bounded pinned-staging asynchronous pipeline; izoluje efekt overlapu.
12. `hybrid_dynamic_fixed` — centralna kolejka ze stałym chunk size.
13. `hybrid_dynamic_guided` — guided self-scheduling.
14. `hybrid_dynamic_adaptive` — scheduler przepustowościowy z EMA i docelowym czasem porcji.

## Dlaczego taki zestaw jest mocniejszy naukowo

- zawiera słabe baseline'y, mocne baseline'y i implementację biblioteczną;
- rozdziela problem „mikro” (jak redukować na pojedynczym urządzeniu) od „makro” (jak dzielić pracę CPU/GPU);
- nie uzależnia podstawowej tezy od arbitralnych wag 1:15;
- nie miesza P2P/NUMA z głównym pytaniem badawczym;
- pozwala przypisać efekt overlapu do polityki transferu, a nie do całkiem innego schedulera;
- pozwala przetestować hipotezę, czy model statyczny, samobalansowanie dynamiczne czy adaptacja przepustowości daje najlepszy kompromis czas/energia.

## Warianty dodatkowe, nie do głównego rankingu

- manual static weight — tylko jako sensitivity analysis;
- distributed work stealing — tylko jeśli central queue okaże się bottleneckiem;
- multi-GPU P2P merge — osobny eksperyment topologiczny;
- resident-data GPU — osobny scenariusz, nie mieszany z end-to-end host-resident.

## Baseline'y skalowania multi-GPU dodane w v2.2

Na serwerach z co najmniej dwoma GPU katalog zawiera również dwa kontrolowane baseline'y, które nie zmieniają głównego zestawu czternastu strategii CPU/GPU, lecz są niezbędne do oceny, czy CPU faktycznie wnosi korzyść względem samego zespołu GPU:

15. `gpu_multi_cub_equal` — CUB na wszystkich wybranych GPU z równym podziałem danych.
16. `gpu_multi_cub_profiled` — CUB na wszystkich wybranych GPU z podziałem wyznaczonym z osobnego modelu kosztu każdego GPU.

Dzięki nim na maszynie 2×GPU można porównać bezpośrednio `GPU0+GPU1` z `CPU+GPU0+GPU1`, zamiast porównywać hybrydę wyłącznie z pojedynczą kartą. Nie należy łączyć tych baseline'ów z algorytmami P2P/NVLink: końcowe skalary są scalane na hoście, dzięki czemu badany jest przede wszystkim podział pracy i przepustowość urządzeń, a nie specyficzna topologia peer-to-peer.
