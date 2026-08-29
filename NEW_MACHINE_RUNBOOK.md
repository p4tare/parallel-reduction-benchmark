# New multi-GPU machine runbook (v2.7.0)

Do not start the confirmatory campaign until the fast smoke passes.

```bash
source venv/bin/activate
python -m pip install -e '.[dev]'
rm -rf build

unset CUDA_VISIBLE_DEVICES
prbench doctor | tee doctor_new_machine.json

prbench preflight --config configs/MULTI_GPU_FULL_STACK_SMOKE_v27.yaml \
  | tee preflight_multi_gpu_smoke.json
prbench plan --config configs/MULTI_GPU_FULL_STACK_SMOKE_v27.yaml \
  > plan_multi_gpu_smoke.json
prbench run --config configs/MULTI_GPU_FULL_STACK_SMOKE_v27.yaml \
  | tee console_multi_gpu_smoke.txt
```

Inspect the newest run: all tasks should be `ok`; `energy_complete_for_requested_components`
should be true when RAPL/NVML are enabled.  Then run preflight/plan for the confirmatory study:

```bash
prbench preflight --config configs/FINAL_MULTI_GPU_RESEARCH_v27.yaml \
  | tee preflight_final_multi_gpu.json
prbench plan --config configs/FINAL_MULTI_GPU_RESEARCH_v27.yaml \
  > plan_final_multi_gpu.json
```

Only if preflight reports `status: ok` and the plan has the intended GPU sets should the
actual run be started:

```bash
prbench run --config configs/FINAL_MULTI_GPU_RESEARCH_v27.yaml \
  | tee console_final_multi_gpu.txt
```

The large-data run has a mandatory preflight because it contains a 64 GB host dataset:

```bash
prbench preflight --config configs/LARGE_DATA_MULTI_GPU_v27.yaml \
  | tee preflight_large_multi_gpu.json
prbench plan --config configs/LARGE_DATA_MULTI_GPU_v27.yaml \
  > plan_large_multi_gpu.json
# run only after inspecting both files
```

Key fields to inspect in `doctor_new_machine.json` / preflight:

- number, names, UUIDs, VRAM total/free/used and PCIe link for every GPU;
- NUMA node and local CPU list per GPU;
- P2P matrix and `nvidia-smi topo -m` in the manifest;
- `gpu_energy_diagnostics` (hardware total-energy counter vs power fallback);
- readable RAPL package zones;
- no foreign compute processes on selected GPUs;
- no non-identity `CUDA_VISIBLE_DEVICES`;
- `gpu_memory_capacity` has no exact over-budget allocations;
- any multi-NUMA warning is documented when interpreting scaling.
