# v2.5

## CUDA build fix

`CUDA_CHECK` is now a variadic macro. v2.4 wrapped calls such as `cub_reduce<T, Op>(...)` in a one-argument macro; the C preprocessor treats the comma in `<T, Op>` as a macro-argument separator, causing NVCC errors `macro "CUDA_CHECK" passed 2 arguments, but takes just 1`. The variadic form safely wraps both ordinary CUDA runtime calls and templated helper calls.

## Build error UX

Native CMake/build failures are converted to a concise `BUILD ERROR:` message after the compiler diagnostics rather than exposing a Python `CalledProcessError` traceback. Compiler output remains visible.

## Administration documentation

Added `ADMIN_REQUIREMENTS.md` and `ADMIN_EMAIL_TEMPLATE_PL.md` describing the required toolchain, RAPL read permissions, NVIDIA/NVML access and topology sysfs paths.
