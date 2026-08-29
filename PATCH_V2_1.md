# v2.1 — coherent CUDA/C++ toolchain and fresh CMake configuration

This patch addresses a portability failure observed when a CPU-only CMake build and a CUDA build use different GCC/libstdc++ generations in the same build directory. A typical symptom is:

```
undefined reference to `__cxa_call_terminate'
```

## Changes

- selects one C++ compiler that passes a C++20 compile probe;
- when CUDA is enabled, additionally probes that compiler through `nvcc -ccbin`;
- uses the same compiler as `CMAKE_CXX_COMPILER` and `CMAKE_CUDA_HOST_COMPILER`;
- passes `CMAKE_CUDA_COMPILER` explicitly;
- configures with `cmake --fresh` to prevent stale CPU/CUDA compiler cache state;
- records exact compiler paths and versions in build metadata;
- adds optional `build.cxx_compiler` and `build.cuda_host_compiler` overrides;
- expands `prbench doctor` with CMake/C++/NVCC toolchain information;
- uses `Threads::Threads` instead of raw `pthread` linkage.

## Recommended retry

Because v2.1 uses `cmake --fresh`, a manual clean is normally unnecessary, but after upgrading from v2 it is safe to run:

```bash
rm -rf build
prbench doctor
prbench run --config configs/smoke_cuda.yaml
```

If automatic selection still fails, specify the same compiler explicitly, e.g.:

```yaml
build:
  cxx_compiler: /usr/bin/g++-13
  cuda_host_compiler: /usr/bin/g++-13
```
