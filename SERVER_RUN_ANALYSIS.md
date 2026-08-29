# Analysis of des07 validation run

## Detected topology

After the native worker was built, CPUID classification is complete:

- logical CPUs 0..15: `performance`, source `native_cpuid`;
- these correspond to 8 physical P-cores with two logical threads each;
- logical CPUs 16..23: `efficiency`, source `native_cpuid`;
- these correspond to 8 physical E-cores with one logical thread each;
- total: 16 physical cores / 24 logical CPUs;
- one NUMA node;
- one NVIDIA GeForce RTX 4070 Ti, CC 8.9, ~12 GiB VRAM;
- RAPL sysfs present and NVML available.

The topology therefore supports P-only, E-only and all-core studies on this server. It does not support multi-GPU or cross-NUMA conclusions; those belong to the second server.

## Failed-task diagnosis

All observed failures have one root cause and are not random instability:

- `hybrid_dynamic_fixed`, `chunk_size=262144` fails before READY;
- `chunk_size=1048576` and `4194304` pass;
- both pilot dataset sizes and both blocks reproduce the 262144 failure;
- stderr: `GPU reducer received more elements than configured max_elements`.

The exact v2.2 source defect was the unconditional 1 Mi-element throughput calibration performed even for the fixed scheduler, while its GPU reducer was allocated for only 262144 elements. v2.3 removes that unnecessary calibration and includes a regression configuration.

## Interpretation

The successful tasks are strong evidence that the common CUDA worker, CUB path, custom GPU backends, CPU backends, P/E affinity resolution, and the other hybrid schedulers execute on this server. They are not yet final thesis data because the run belongs to validation/tuning and one code defect was subsequently fixed. The complete pilot should be rerun on v2.3 so every tuning choice is made from one source revision and one measurement protocol.
