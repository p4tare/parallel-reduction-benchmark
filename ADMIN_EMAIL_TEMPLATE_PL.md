Temat: Prośba o przygotowanie środowiska do pomiarów CPU/GPU i udostępnienie odczytu Intel RAPL

Dzień dobry,

w ramach pracy magisterskiej wykonuję pomiary wydajności i energii algorytmów redukcji na CPU oraz NVIDIA GPU. Chciałbym prosić o przygotowanie/utrzymanie poniższego środowiska na serwerze obliczeniowym oraz o nadanie mojego konta/grupy badawczej uprawnień wyłącznie do odczytu liczników energii.

Wymagane elementy środowiska:
- Linux x86-64;
- Python >= 3.10 wraz z venv i pip;
- CMake >= 3.24 oraz GNU Make;
- GCC/G++ 13 z obsługą C++20 i OpenMP (na obecnym serwerze CUDA Toolkit 12.4 korzysta z G++ 13 jako host compilera);
- runtime OpenMP/libgomp i pthread;
- sterownik NVIDIA, nvidia-smi i biblioteka NVML (libnvidia-ml.so.1);
- CUDA Toolkit 12.4.x wraz z nvcc, CUDA runtime i nagłówkami CUB;
- numactl oraz lscpu (zalecane do kontroli NUMA i zapisu metadanych środowiska);
- Git (zalecany do jednoznacznego oznaczania wersji użytej w badaniu);
- opcjonalnie lm-sensors / działający kernelowy hwmon dla kontroli temperatur CPU.

Do pomiaru energii CPU program korzysta z Linux powercap / Intel RAPL. Proszę o zapewnienie działania odpowiedniego sterownika RAPL oraz uprawnienia do odczytu dla mojego użytkownika lub dedykowanej grupy do package-level plików:
/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj
/sys/class/powercap/intel-rapl/intel-rapl:*/max_energy_range_uj
/sys/class/powercap/intel-rapl/intel-rapl:*/name
oraz prawa przejścia (x) przez odpowiadające katalogi.

Na obecnym serwerze licznik istnieje, ale `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj` ma uprawnienia `0400 root:root`, przez co zwykły użytkownik dostaje `Permission denied`. Nie potrzebuję prawa zapisu do liczników ani limitów mocy. Preferowane byłoby trwałe rozwiązanie przez grupę/ACL/udev/systemd-tmpfiles, zamiast uruchamiania programu przez sudo.

Do pomiaru GPU program korzysta z NVML, dlatego konto musi mieć standardowy dostęp do urządzeń NVIDIA (`/dev/nvidiactl`, `/dev/nvidia0`, ewentualnie kolejnych GPU i `/dev/nvidia-uvm`) oraz do biblioteki NVML. Na obecnym serwerze NVML jest dostępne.

Dodatkowo program odczytuje zwykłe metadane topologii z `/sys/devices/system/cpu/...` i `/sys/bus/pci/devices/...`; wystarczają standardowe prawa odczytu.

Jeśli to możliwe, zależy mi również na możliwości wykonywania finalnych pomiarów przy wyłącznym dostępie do noda/GPU, ponieważ Intel RAPL mierzy energię całego pakietu CPU, a obce procesy zafałszowują wynik.

Z góry dziękuję.
