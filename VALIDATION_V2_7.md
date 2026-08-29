# Validation v2.7.0

Validation performed in the preparation environment (no NVIDIA GPU/NVCC available locally):

- Python `compileall`: PASS
- `pytest`: **34/34 PASS**
- CMake Release CPU-only configure/build: PASS
- native `ctest`: **1/1 PASS**
- synthetic planner validation for 2 GPUs: PASS
- synthetic planner validation for 3 GPUs: PASS
- `FINAL_MULTI_GPU_RESEARCH_v27.yaml`: validates and plans on synthetic 2/3-GPU topologies
- `LARGE_DATA_MULTI_GPU_v27.yaml`: validates and plans on synthetic 2/3-GPU topologies
- `MULTI_GPU_FULL_STACK_SMOKE_v27.yaml`: validates and plans on synthetic 2/3-GPU topologies

Synthetic plan sizes:

| config | 2 GPUs | 3 GPUs |
|---|---:|---:|
| multi-GPU smoke | 45 task instances | 156 task instances |
| confirmatory multi-GPU study | 810 | 1920 |
| large-data extension | 153 | 369 |

CUDA/multi-GPU runtime cannot be certified in the preparation environment.  Before thesis data are collected on the new machine, `doctor`, `preflight`, and `MULTI_GPU_FULL_STACK_SMOKE_v27.yaml` must pass on the actual hardware.  In particular verify CUDA/NVML index identity, RAPL access, per-GPU energy method, VRAM capacity, PCIe/NUMA topology and absence of foreign GPU compute processes.
