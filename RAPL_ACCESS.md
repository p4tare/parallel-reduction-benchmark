# RAPL access for research measurements

`rapl_present` and `rapl_available` mean different things:

- `rapl_present=true`: the Linux powercap Intel RAPL sysfs hierarchy exists;
- `rapl_available=true`: at least one package-level `energy_uj` counter and its `max_energy_range_uj` are actually readable by the current user.

The second condition is required for CPU and hybrid energy-to-solution measurements.

## Diagnose

With v2.4 run:

```bash
prbench doctor > doctor.json
prbench preflight --config configs/energy_validation_single_gpu.yaml
```

Both commands include `rapl_diagnostics`. You can also inspect sysfs directly:

```bash
find /sys/class/powercap/intel-rapl -maxdepth 2 \
  -type f \( -name energy_uj -o -name max_energy_range_uj -o -name name \) \
  -exec ls -l {} \;

for f in /sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj; do
  echo "=== $f ==="
  cat "$f"
done
```

If `cat` returns `Permission denied`, the measurement code cannot solve this in user space.

## Recommended administrator request

Ask the server administrator to grant the benchmark user or a dedicated research group read access to the package-level RAPL files:

```text
/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj
/sys/class/powercap/intel-rapl/intel-rapl:*/max_energy_range_uj
/sys/class/powercap/intel-rapl/intel-rapl:*/name
```

Prefer a group/ACL/udev or equivalent persistent policy appropriate for the server. Do not run the complete benchmark as root merely to bypass the permission problem, and do not use world-writable permissions.

After access is granted, verify that an unprivileged `cat energy_uj` works and rerun `prbench preflight`.

## What can be measured without RAPL

With:

```yaml
energy:
  enable_cpu: false
  enable_gpu: true
```

GPU energy can still be validated and measured through NVML. This is useful for GPU-only experiments, but it is **not** sufficient to claim total energy-to-solution for CPU or CPU+GPU strategies because CPU package energy is missing.

If RAPL access cannot be obtained, a defensible alternative is an external whole-node energy/power meter. That changes the measurement boundary and should be treated as a separate methodology rather than mixed silently with RAPL/NVML results.
