# GPU break-even / memory-path study

This branch is intentionally isolated from the frozen thesis-result branch point
`ce2ff9dc3c3438448389af5ac698be329758add5`. Existing final configurations and result
bundles are not modified.

## Research question

The new experiment asks when a host-resident reduction becomes worth offloading to a
discrete GPU after including data movement, and which memory/data-placement policy moves
that break-even point.

The main host-resident experiment keeps `dataset <= host RAM`. Out-of-core storage is a
separate diagnostic so NVMe/filesystem effects cannot be confused with CPU/GPU reduction.

## Implemented experimental paths

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

## GPUDirect Storage

When CMake finds both `cufile.h` and `libcufile`, it builds `prbench-gds-worker`. The worker
opens the dataset with `O_DIRECT`, registers the file with cuFile, reads each chunk directly
into a CUDA device buffer with `cuFileRead`, reduces it with CUB, and combines scalar partials.
It is deliberately separate from host-resident ranking.

If cuFile is absent, the main project still builds and `prbench-memory-paths doctor` reports
GDS prerequisites instead of silently substituting a host bounce-buffer path.

## Python orchestration

Install the branch editable and build with CUDA, then run:

```bash
python -m pip install -e '.[dev]'
cmake -S . -B build --fresh -DPRBENCH_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

prbench-memory-paths doctor
prbench-memory-paths run configs/experimental/memory_paths_smoke.yaml
```

The orchestrator writes:

- `memory_path_manifest.json` - full config, host RAM, HMM and GDS diagnostics;
- `memory_path_repetitions.jsonl` - append-only raw task results;
- `memory_path_summary.csv` - flat analysis input.

Tasks are randomized independently per block using `randomization_seed + block`.

## apl13 sequence

Do **not** start with the 112-GiB sweep. The intended sequence is:

1. build the branch on apl13;
2. run `prbench-memory-paths doctor`;
3. run `configs/experimental/memory_paths_smoke.yaml`;
4. inspect correctness, allocation failures, HMM availability, pinned-memory limits, and GDS;
5. tune chunk size/stream count using `apl13_break_even_tuning.yaml` on a reduced subset;
6. freeze tuning parameters before examining confirmatory break-even results;
7. run the reuse-count study separately;
8. run GDS only as an out-of-core/storage experiment if cuFile and storage topology pass
   preflight.

## Interpretation

For a one-pass reduction the GPU is normally bandwidth-bound. Large data does not guarantee
a GPU win. The relevant asymptotic comparison is the effective end-to-end byte rate:

`CPU RAM -> CPU reduction`

versus

`host RAM -> interconnect -> GPU + GPU reduction`.

A crossing is a measured result, not an assumption. If the slopes do not cross, the study
should report that no single-use break-even exists in the tested host-resident range and then
quantify the `reuse_count` needed to amortize placement.

## Known validation requirements before thesis data

The CUDA workers must be compiled and smoke-tested on the actual NVIDIA host. GitHub CI is
CPU-only and can validate the Python planner plus the pre-existing CPU worker, but it cannot
validate CUDA/HMM/cuFile behavior. Large mapped/pinned allocations may also be rejected by OS
or driver limits; those failures are experimental capability results and must not be hidden.
