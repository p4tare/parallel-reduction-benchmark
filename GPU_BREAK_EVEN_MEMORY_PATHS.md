# GPU break-even / memory-path study

This branch is intentionally isolated from the frozen thesis-result branch point
`ce2ff9dc3c3438448389af5ac698be329758add5`. Existing final configurations and result
bundles are not modified.

## Research question

The new experiment asks when a reduction becomes worth offloading to a discrete GPU after
including data movement, which memory/data-placement policy moves that break-even point, and
how the result changes when several GPUs and the CPU operate concurrently.

There is no global `dataset <= RAM` restriction. Results are explicitly tagged as either
host-resident or out-of-core/file-stream and must not be pooled into a single ranking.

## Implemented single-GPU memory paths

`prbench-memory-path-worker` fixes the reduction implementation to CUB and varies only the
memory path:

- `explicit_sync` - full H2D placement, CUB reduction, scalar D2H;
- `chunked_sync` - bounded VRAM buffer; every input byte is transferred once in chunks;
- `async_pipeline` - pinned staging buffers + multiple streams, overlapping H2D/reduction;
- `zero_copy` - `cudaHostAllocMapped`, CUB reads mapped host memory over the interconnect;
- `managed_fault` - `cudaMallocManaged`, demand-fault/page migration controlled by CUDA;
- `managed_prefetch` - managed allocation plus `cudaMemPrefetchAsync`;
- `managed_advised` - managed allocation plus read-mostly/accessed-by advice. This mode may
  preserve or duplicate read-mostly pages and is therefore a placement/reuse policy rather
  than a pure first-touch migration benchmark;
- `hmm_system` - ordinary system allocation passed to the GPU when
  `cudaDevAttrPageableMemoryAccess` confirms system/HMM access;
- `device_resident` - diagnostic residency endpoint used to estimate the reduction-only
  lower bound;
- `reuse_count` is orthogonal and repeats the reduction on the same placement to study
  transfer amortization.

## Out-of-core paths

`prbench-file-stream-worker` supports `chunked_sync` and `async_pipeline` without loading the
whole dataset into system RAM. It uses bounded pinned host staging buffers and records storage
read time/bytes separately from H2D and reduction. Therefore datasets may exceed host RAM,
subject to storage capacity and filesystem performance.

Host-resident and file-stream results answer different questions and are tagged separately.

## Topology streaming paths

`prbench-topology-stream-worker` adds the remaining transport/topology mechanisms:

- `pinned_direct` - register the already populated host dataset with `cudaHostRegister` and
  transfer chunks directly from that allocation, removing the extra pageable->pinned staging
  memcpy used by the ordinary async pipeline;
- `multi_gpu_async` - partition one dataset across several GPUs and stream the partitions
  concurrently, with independent streams and CUB reductions per GPU;
- `hybrid_cpu_gpu` - reduce one partition on the CPU while the remaining partitions are
  streamed/reduced concurrently on one or more GPUs;
- NUMA-local registered staging - when libnuma is present, each GPU worker can allocate its
  registered staging buffers on an explicitly configured NUMA node and bind the worker to
  that node;
- both `multi_gpu_async` and `hybrid_cpu_gpu` support `host_resident` and `file_stream`, so the
  same topology can be studied below and above host-RAM capacity.

The topology worker is an experimental transport/topology study. For the final host-resident
hybrid ranking, the pre-existing main benchmark remains the reference because it uses the
established CPU backends and schedulers. The topology worker is primarily used to isolate
multi-device streaming, NUMA placement and >RAM behavior before confirmatory configuration is
frozen.

## GPUDirect Storage

When CMake finds both `cufile.h` and `libcufile`, it builds `prbench-gds-worker`. The worker
opens the dataset with `O_DIRECT`, registers the file with cuFile, reads each chunk directly
into a CUDA device buffer with `cuFileRead`, reduces it with CUB, and combines scalar partials.
It is deliberately separate from host-resident ranking.

If cuFile is absent, the main project still builds and `prbench-memory-paths doctor` reports
GDS prerequisites instead of silently substituting a host bounce-buffer path.

## apl13 validated capabilities (2026-09-16 smoke)

The first memory-path smoke run contained 132 tasks: 120 completed and validated correctly,
0 produced an incorrect numerical result, and 12 were intentionally skipped. All skips were
`hmm_system`.

The two Quadro RTX 8000 devices reported:

- managed memory: available;
- concurrent managed access: available;
- mapped host memory / zero-copy: available;
- pageable GPU access: unavailable;
- host page-table access: unavailable;
- `nvidia-smi` addressing mode: `None` on both devices.

Consequently HMM is not available on the tested apl13 driver/kernel/GPU configuration and must
remain a capability-gated skip. This is a machine capability result, not a benchmark failure.

The same machine did not expose `libcufile`, `gdscheck`, `/dev/nvidia-fs`, or a loaded
`nvidia_fs` module. GPUDirect Storage therefore remains implemented but unavailable on apl13
unless the system software/storage stack is changed.

## Python orchestration

Build with one consistent host compiler for C++ and NVCC on apl13:

```bash
python -m pip install -e '.[dev]'
cmake -S . -B build --fresh \
  -DCMAKE_BUILD_TYPE=Release \
  -DPRBENCH_ENABLE_CUDA=ON \
  -DPRBENCH_ENABLE_GDS=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build --parallel
```

Single-GPU memory-path smoke:

```bash
prbench-memory-paths doctor
prbench-memory-paths run configs/experimental/memory_paths_smoke.yaml
```

Topology/multi-device smoke:

```bash
prbench-topology-streams configs/experimental/topology_stream_smoke.yaml
```

The memory-path orchestrator writes `memory_path_manifest.json`,
`memory_path_repetitions.jsonl`, and `memory_path_summary.csv`. The topology-stream orchestrator
writes an independent `topology_stream_manifest.json`, `topology_stream_repetitions.jsonl`,
and `topology_stream_summary.csv`.

Tasks are randomized independently per block using `randomization_seed + block`.

## Intended experimental sequence

Do not start with the largest dataset immediately. The sequence is:

1. compile and run the single-GPU memory-path smoke;
2. validate capabilities and numerical correctness;
3. compile and run `topology_stream_smoke.yaml` for pinned-direct, multi-GPU, hybrid and NUMA;
4. tune chunk size and stream count on a reduced size subset;
5. tune CPU/GPU partition for the hybrid path without using confirmatory results;
6. freeze tuning parameters;
7. run the host-resident break-even sweep through and beyond one-GPU VRAM;
8. run the reuse-count study separately;
9. run the out-of-core/file-stream study for datasets beyond RAM;
10. run GDS only if a future apl13 configuration exposes cuFile and the required storage stack.

## Interpretation

For a one-pass reduction the GPU is normally bandwidth-bound. Large data does not guarantee
a GPU win. The relevant asymptotic comparison is the effective end-to-end byte rate:

`CPU RAM -> CPU reduction`

versus

`host RAM -> interconnect -> GPU + GPU reduction`.

For several devices the experiment additionally asks whether concurrent CPU memory access and
one/two DMA streams increase aggregate useful throughput or instead saturate the same host
memory/PCIe bottleneck.

A crossing is a measured result, not an assumption. If the slopes do not cross, the study
should report that no single-use break-even exists in the tested range and then quantify the
`reuse_count` needed to amortize placement.

## Known validation requirements before thesis data

CUDA/topology workers must be compiled and smoke-tested on the actual NVIDIA host. GitHub CI
is CPU-only and cannot validate CUDA, NUMA, HMM or cuFile behavior. Large mapped/registered
allocations may also be rejected by OS/driver resource limits; those failures are capability
results and must not be hidden.
