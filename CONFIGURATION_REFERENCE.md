# Pełna referencja konfiguracji v2.6

Najbardziej praktycznym źródłem jest `configs/CONFIG_ALL_OPTIONS_TEMPLATE.yaml`: to poprawny YAML z komentarzami, wszystkimi wartościami kategorialnymi i przykładowymi siatkami parametrów.

## Build

- `build_type`: `Release` | `RelWithDebInfo`.
- `native_cpu_tuning`: `true` | `false`.
- `enable_cuda`: `auto` | `on` | `off`.
- `jobs`: brak/null albo dodatni int.
- `cxx_compiler`, `cuda_host_compiler`: opcjonalne ścieżki.

## Measurement

`timing_repetitions`:
- `auto` (rekomendowane) — warm-up -> niewliczany PROBE -> batch o docelowym czasie;
- albo stały int 3..1,000,000.

Parametry auto TIMING:
- `timing_probe_repetitions`: 3..1000;
- `timing_target_batch_seconds`: >0.01 .. 30;
- `timing_min_repetitions`: >=3;
- `timing_max_repetitions`: >=min, <=1,000,000.

ENERGY:
- `energy_batch_repetitions`: `auto` lub stały int w zakresie min/max;
- `energy_target_batch_seconds`: >0.1 .. 120;
- `energy_min_repetitions`, `energy_max_repetitions`: dodatnie.

Pozostałe:
- `warmup_runs` >=1;
- `scheduler_calibration_repetitions`: 1..100, mediana per punkt modelu;
- `blocks`: 1..100;
- `randomization_seed`: int;
- `max_dataset_ram_fraction`: >0.05 .. 0.95;
- `thermal_safety_gpu_c`: 40..110;
- `thermal_safety_cpu_c`: 40..115;
- `thermal_wait_timeout_s` >=0;
- `worker_event_timeout_s`: >0 .. 86400.

## Energia

- `enable_cpu`: bool;
- `enable_gpu`: bool;
- `gpu_power_fallback_poll_ms`: 20..5000.

Semantyka v2.6:
- brak mierzonego komponentu -> brak bezsensownego ENERGY batcha;
- CPU RAPL jest mierzone dla każdego tasku, gdy `enable_cpu=true`, w tym GPU-only;
- GPU energia tylko dla faktycznie używanych GPU, gdy `enable_gpu=true`;
- każdy rekord ma `energy_coverage`.

## Telemetria

```yaml
telemetry:
  enabled: true
  capture_cpu_frequency: true
  capture_cpu_temperature: true
  capture_gpu_state: true
  capture_pre_post_timing: true
  capture_pre_post_energy: true
```

Snapshoty są wykonywane poza TIMING/ENERGY i trafiają do `telemetry.jsonl`.

## System baselines

```yaml
system_baselines:
  capture_memory_inventory: true
  stream_executable: null
  stream_args: []
  stream_timeout_s: 120
```

STREAM jest opcjonalnym zewnętrznym executable; standardowe Copy/Scale/Add/Triad są parsowane do manifestu.

## Operacje

`sum` | `min` | `max` — dowolny niepusty podzbiór per experiment group.

## Datasets

`dtype`: `int32` | `int64` | `float32` | `float64`.

`distribution`: `ones` | `zeros` | `uniform` | `integers` | `symmetric_uniform`.

`quantization.mode`: `none` | `decimal` | `binary_fraction`.
- decimal: `digits` 0..12;
- binary_fraction: `bits` 0..24;
- kwantyzacja tylko float.

`size`: dodatni int, praktycznie ograniczony RAM i `max_dataset_ram_fraction`.

## CPU / GPU / NUMA

- `cpu_core_class`: `all` | `performance` | `efficiency`;
- `cpu_thread_policy`: `all_threads` | `one_thread_per_core`;
- `cpu_numa_node`: null lub >=0;
- `cpu_explicit_ids`: opcjonalna niepusta unikalna lista OS CPU IDs, override filtrów automatycznych;
- `gpu_control_mode`: `dedicated` | `shared`;
- `memory_policy`: `default` | `interleave`;
- `gpu_sets`: `each`, `all`, pojedynczy indeks, jawna lista indeksów.

Każdy GPU dostaje odrębny fizyczny control core, z preferencją lokalności PCIe/NUMA.

## Algorytmy

CPU:
- `cpu_seq`
- `cpu_omp`
- `cpu_omp_simd`

GPU single:
- `gpu_global_atomic`: `block_size`
- `gpu_shared_naive`: `block_size`
- `gpu_warp_atomic`: `block_size`
- `gpu_two_pass`: `block_size`
- `gpu_cub`

`block_size`: `32, 64, 128, 256, 512, 1024`.

Multi-GPU:
- `gpu_multi_cub_equal` (>=2 GPU)
- `gpu_multi_cub_profiled` (>=2 GPU)

Hybrid static:
- `hybrid_static_equal`
- `hybrid_static_profiled`
- `hybrid_static_profiled_async`: dodatnie `pipeline_streams`, `pipeline_chunks`, z `chunks >= streams`.

Hybrid dynamic:
- `hybrid_dynamic_fixed`: dodatni `chunk_size`;
- `hybrid_dynamic_guided`: dodatni `min_chunk_size`, `guided_factor > 0`;
- `hybrid_dynamic_adaptive`: dodatnie min/max chunk (`max>=min`), `target_chunk_ms>0`, `0<ema_alpha<=1`.

Parametry liczbowe mają nieskończone dziedziny; `CONFIG_ALL_OPTIONS_TEMPLATE.yaml` zawiera rozsądną szeroką siatkę do pilota/sensitivity analysis.

## Projekt eksperymentu

Finalna kampania powinna używać symetrycznych grup. Jeśli oceniasz P/E/all, identyczne muszą pozostać operacje, datasety, algorytmy, GPU-control mode i parametry. Analogicznie dla dedicated/shared. `preflight` wykrywa część oczywistych niespójności i emituje warnings, ale nie zastępuje świadomego projektu eksperymentu.


## Multi-GPU selectors (v2.7.0)

`hardware.gpu_sets` supports:

- `each` -- one task for every visible GPU;
- `pairs` / `all_pairs` -- every two-GPU combination (only for multi-GPU-capable algorithms);
- `all` -- all visible GPUs;
- an integer such as `0` -- one explicit GPU;
- an explicit list such as `[0, 2]` -- one explicit GPU set.

`measurement.gpu_memory_safety_fraction` (default `0.80`) reserves VRAM headroom for
CUB temporary storage, CUDA context/runtime allocations and other driver use. Preflight
fails only for capacity estimates known exactly; profile-dependent partitions are reported
as warnings.
