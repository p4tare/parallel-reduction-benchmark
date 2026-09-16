# GPU break-even / memory-path study

This branch is intentionally isolated from the frozen thesis-result branch point
`ce2ff9dc3c3438448389af5ac698be329758add5`. Existing final configurations and result
bundles are not modified.

## Research question

The experiment asks when reduction becomes worth offloading to a discrete GPU after
including data movement, and which memory/data-placement policy moves that break-even point.

There is no global `dataset <= host RAM` restriction. Two execution classes are kept
explicitly separate:

- `host_resident`: the whole input is resident in system RAM before the measured reduction;
- `file_stream`: a bounded pinned host buffer streams the dataset from the backing file to
  the GPU, so the logical dataset may be larger than RAM.

These classes are tagged in every result and must not be pooled into one performance ranking,
because the file-stream path includes storage/filesystem effects.

## Implemented host-resident paths

`prbench-memory-path-worker` fixes the reduction implementation to CUB and varies only the
memory path:

- `explicit_sync` - full H2D placement, CUB reduction, scalar D2H;
- `chunked_sync` - bounded VRAM buffer; every input byte is transferred once in chunks;
- `async_pipeline` - pinned staging buffers + multiple streams, overlapping H2D/reduction;
- `zero_copy` - `cudaHostAllocMapped`, CUB reads mapped host memory over the interconnect;
- `managed_fault` - `cudaMallocManaged`, demand-fault/page migration controlled by CUDA;
- `managed_prefetch` - managed allocation plus `cudaMemPrefetchAsync`;
- `managed_advised` - managed allocation plus `cudaMemAdviseSetReadMostly` and
  `cudaMemAdviseSetAccessedBy`;
- `hmm_system` - ordinary system allocation passed to the GPU when
  `cudaDevAttrPageableMemoryAccess` confirms system/HMM access;
- `device_resident` - diagnostic residency endpoint used to estimate the reduction-only
  lower bound;
- `reuse_count` is orthogonal and repeats the reduction on the same placement to study
  transfer amortization.

The worker reports CUDA capability attributes with every result, including managed memory,
concurrent managed access, pageable memory access, host page tables, and mapped-host support.

## Datasets larger than RAM

`prbench-file-stream-worker` implements a bounded-memory out-of-core path. It supports:

- `chunked_sync`: file -> pinned host chunk -> H2D -> CUB -> scalar partial;
- `async_pipeline`: file -> bounded pinned buffers -> multiple CUDA streams -> CUB.

The worker never allocates the complete dataset in RAM. Therefore the logical dataset may be
larger than both GPU VRAM and system RAM, provided the backing filesystem has enough space.
The result records `storage_policy=file_stream`, storage bytes and storage read time.

The Python orchestrator treats `host_resident_ram_fraction` only as a routing boundary, not
as a dataset-size prohibition. Above that boundary:

- `chunked_sync` and `async_pipeline` automatically use `file_stream`;
- full-residency modes (`explicit_sync`, `zero_copy`, Managed Memory variants, HMM system
  allocation and `device_resident`) are recorded as `skipped` with an explicit reason.

This prevents accidental swapping or hidden overcommit from being mistaken for a valid
host-resident result.

## GPUDirect Storage

When CMake finds both `cufile.h` and `libcufile`, it builds `prbench-gds-worker`. The worker
opens the dataset with `O_DIRECT`, registers the file with cuFile, reads each chunk directly
into a CUDA device buffer with `cuFileRead`, reduces it with CUB, and combines scalar partials.
It is deliberately separate from host-resident ranking.

If cuFile is absent, the main project still builds and `prbench-memory-paths doctor` reports
GDS prerequisites instead of silently substituting a host bounce-buffer path.

## apl13 compiler requirement

The apl13 image currently exposes GCC/G++ 15 as the default C++ compiler while CUDA 12.4
uses a GCC 13 host compiler. Mixing those compilers in `prbench-worker` can fail at link time
with unresolved C++ ABI symbols such as `__cxa_call_terminate`.

CMake now rejects a mismatched C++/NVCC host compiler early. Configure apl13 explicitly with
G++ 13 for both languages:

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPRBENCH_ENABLE_CUDA=ON \
  -DPRBENCH_ENABLE_GDS=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build --parallel
```

## Python orchestration

Install the branch editable, build with CUDA, then run:

```bash
python -m pip install -e '.[dev]'
prbench-memory-paths doctor
prbench-memory-paths run configs/experimental/memory_paths_smoke.yaml
```

The orchestrator writes:

- `memory_path_manifest.json` - config, RAM boundary, HMM/GDS diagnostics and storage-policy
  definitions;
- `memory_path_repetitions.jsonl` - append-only raw task results;
- `memory_path_summary.csv` - flat analysis input including `storage_policy`.

Tasks are randomized independently per block using `randomization_seed + block`.

## apl13 sequence

Do **not** start with the largest sweep. The intended sequence is:

1. build the branch on apl13 with the consistent GCC 13 toolchain;
2. run `prbench-memory-paths doctor`;
3. run `configs/experimental/memory_paths_smoke.yaml`;
4. inspect correctness, allocation failures, HMM availability, pinned-memory limits, and GDS;
5. tune chunk size/stream count using `apl13_break_even_tuning.yaml` on a reduced subset;
6. freeze tuning parameters before confirmatory break-even measurements;
7. run the reuse-count study separately;
8. extend beyond host RAM only after measuring the backing-storage baseline; treat those
   results as out-of-core rather than host-resident;
9. run GDS only if cuFile and storage topology pass preflight.

## Interpretation

For a one-pass host-resident reduction the relevant asymptotic comparison is:

`CPU RAM -> CPU reduction`

versus

`host RAM -> interconnect -> GPU + GPU reduction`.

For datasets larger than RAM the comparison changes and must include storage:

`storage -> bounded host buffer -> CPU/GPU`.

Therefore host-resident and out-of-core break-even points answer different research
questions. A crossing is a measured result, not an assumption.

## Known validation requirements before thesis data

The CUDA workers must be compiled and smoke-tested on the actual NVIDIA host. GitHub CI is
CPU-only and cannot validate CUDA/HMM/cuFile behavior. Large mapped/pinned allocations may
also be rejected by OS or driver limits; those failures are capability results and must not
be hidden.
