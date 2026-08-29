# Administrator requirements for Parallel Reduction Benchmark v2.5

This file lists the operating-system/toolchain facilities expected by the benchmark. The reference research platform is Linux x86-64 with Intel CPU(s) and NVIDIA CUDA GPU(s).

## Required toolchain/runtime

- Python 3.10 or newer, with `venv` and `pip`.
- CMake 3.24 or newer.
- GNU Make (the default CMake generator used by the benchmark; Ninja can be used only if the build configuration is changed).
- GNU C/C++ compiler with C++20 and OpenMP support. For the currently installed CUDA Toolkit 12.4, keep GCC/G++ 13 available and use it as both the normal C++ compiler and NVCC host compiler. The benchmark already auto-selects a compatible compiler.
- POSIX threads (provided by glibc/pthread on Linux).
- OpenMP runtime for GCC (`libgomp`).
- NVIDIA driver supporting the installed GPU(s), CUDA runtime and NVML.
- NVIDIA CUDA Toolkit including `nvcc`, CUDA runtime headers/libraries and CUB headers. The currently verified server has CUDA 12.4.131.
- `numactl` is strongly recommended for reproducible NUMA/interleaved-memory experiments.
- `util-linux`/`lscpu` and `nvidia-smi` are strongly recommended for environment capture in the run manifest.
- Git is strongly recommended so every final run can record an immutable source revision.
- `lm-sensors` is optional for administration/diagnostics; the benchmark itself reads temperatures through psutil/kernel hwmon rather than calling the `sensors` command.

Python packages are installed into the user's virtual environment by `python -m pip install -e '.[dev]'`; they do not require system-wide Python installation beyond Python/venv/pip. The project dependencies are NumPy, psutil, PyYAML, Pydantic and nvidia-ml-py; pytest/coverage/ruff/mypy are development/test dependencies.

## CPU energy / Intel RAPL

The kernel must expose Intel RAPL through Linux powercap, normally under:

`/sys/class/powercap/intel-rapl/`

The benchmark currently measures package-level CPU energy. For every package zone `intel-rapl:N`, the benchmark user needs read/traverse access to the directory and read access to:

- `/sys/class/powercap/intel-rapl/intel-rapl:N/energy_uj`
- `/sys/class/powercap/intel-rapl/intel-rapl:N/max_energy_range_uj`
- `/sys/class/powercap/intel-rapl/intel-rapl:N/name`

On the currently tested server, `energy_uj` exists but is `0400 root:root`, so the user receives `Permission denied`; `max_energy_range_uj` is already readable.

If future measurements need per-domain RAPL values (for example DRAM/core domains), read access should additionally be granted to the same files in nested zones such as `intel-rapl:N:M`. The current primary metric deliberately uses package-level energy.

A persistent group/ACL/udev/systemd-tmpfiles solution is preferred to running the benchmark as root. Only read permission is needed; write permission to power limits or counters is not required.

Useful kernel modules/drivers on Intel systems include the platform's RAPL/powercap drivers (commonly `intel_rapl_common` plus an appropriate Intel RAPL transport such as `intel_rapl_msr`). Exact module names depend on kernel/platform.

## GPU execution and energy / NVML

The benchmark uses NVML rather than a GPU energy sysfs file. The user therefore needs:

- a working NVIDIA kernel driver,
- access to `libnvidia-ml.so.1`,
- access to `nvidia-smi`,
- permissions for the NVIDIA device nodes required by CUDA/NVML, normally `/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia1`, ... and `/dev/nvidia-uvm` where created/required.

NVML must allow `nvmlDeviceGetPowerUsage`; where supported, the benchmark prefers the total-energy counter and otherwise integrates timestamped power samples.

## Topology/sysfs read access

For CPU/NUMA/PCI locality and reproducibility metadata, normal users should retain read/traverse access to:

- `/sys/devices/system/cpu/cpu*/topology/physical_package_id`
- `/sys/devices/system/cpu/cpu*/topology/core_id`
- `/sys/devices/system/cpu/cpu*/online` (where present)
- `/sys/devices/system/cpu/cpu*/node*`
- `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` (where present)
- `/sys/devices/system/cpu/cpufreq/boost` (where present)
- `/sys/devices/system/cpu/intel_pstate/no_turbo` (where present)
- `/sys/bus/pci/devices/<GPU-BDF>/numa_node`
- `/sys/bus/pci/devices/<GPU-BDF>/local_cpulist`

For CPU thermal safety, normal read access to `/sys/class/hwmon/hwmon*/temp*_input` (where exposed by the kernel, e.g. through `coretemp`) is also recommended.

These are read-only requirements for the benchmark. It does not need permission to change governors, turbo state or power limits.

## Recommended reproducibility policy

- Give the student exclusive access to the benchmark node/GPU during final energy measurements where possible.
- Keep the same driver/toolkit/compiler stack during one campaign.
- Do not run the benchmark with `sudo`; grant narrowly scoped read access instead.
