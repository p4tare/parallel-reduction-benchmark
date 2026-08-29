from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import BuildConfig, SystemTopologyModel
from .utils import command_output




class BuildError(RuntimeError):
    """A concise, user-facing native build failure."""


def _run_checked(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise BuildError(f"required build tool was not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(command)
        raise BuildError(
            f"native build command failed with exit code {exc.returncode}: {rendered}. "
            "The compiler diagnostics printed immediately above are the primary error."
        ) from exc


@dataclass(frozen=True)
class BuildArtifact:
    worker_path: Path
    cuda_enabled: bool
    metadata: dict[str, object]


@dataclass(frozen=True)
class ResolvedToolchain:
    cmake: str
    cxx: str
    nvcc: str | None
    cuda_enabled: bool


class CMakeBuilder:
    """Build the generic worker with one coherent host toolchain.

    A mixed GCC/libstdc++ toolchain can produce ABI-level link failures such as
    ``undefined reference to __cxa_call_terminate``.  Therefore, when CUDA is
    enabled, the same C++ compiler is explicitly used for ordinary C++ sources
    and as NVCC's host compiler.  CMake is configured with ``--fresh`` so a
    previous CPU-only cache cannot silently retain a different compiler.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    @staticmethod
    def _resolve_program(value: str | None) -> str | None:
        if not value:
            return None
        candidate = Path(value).expanduser()
        if candidate.parent != Path(".") or "/" in value:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
            return None
        found = shutil.which(value)
        return str(Path(found).resolve()) if found else None

    @staticmethod
    def _compiler_candidates(config: BuildConfig) -> list[str]:
        requested = [
            config.cuda_host_compiler,
            config.cxx_compiler,
            os.environ.get("PRBENCH_CUDA_HOST_COMPILER"),
            os.environ.get("PRBENCH_CXX_COMPILER"),
            os.environ.get("CXX"),
        ]
        # Prefer GNU because NVIDIA's Linux toolchain is most commonly paired
        # with libstdc++; versioned names are useful on clusters with several GCCs.
        requested += [f"g++-{major}" for major in range(16, 5, -1)]
        requested += ["g++", "c++", "clang++"]

        result: list[str] = []
        seen: set[str] = set()
        for item in requested:
            resolved = CMakeBuilder._resolve_program(item)
            if resolved and resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
        return result

    @staticmethod
    def _probe_cxx20(cxx: str) -> tuple[bool, str]:
        source = "#include <thread>\nint main(){ std::thread t([]{}); t.join(); return 0; }\n"
        with tempfile.TemporaryDirectory(prefix="prbench-cxx-probe-") as td:
            src = Path(td) / "probe.cpp"
            obj = Path(td) / "probe.o"
            src.write_text(source, encoding="utf-8")
            proc = subprocess.run(
                [cxx, "-std=c++20", "-c", str(src), "-o", str(obj)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return proc.returncode == 0, proc.stdout.strip()

    @staticmethod
    def _probe_nvcc_host(nvcc: str, cxx: str) -> tuple[bool, str]:
        source = "__global__ void k() {}\nint main(){ return 0; }\n"
        with tempfile.TemporaryDirectory(prefix="prbench-cuda-probe-") as td:
            src = Path(td) / "probe.cu"
            obj = Path(td) / "probe.o"
            src.write_text(source, encoding="utf-8")
            proc = subprocess.run(
                [nvcc, "-ccbin", cxx, "-std=c++17", "-c", str(src), "-o", str(obj)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return proc.returncode == 0, proc.stdout.strip()

    def resolve_toolchain(self, config: BuildConfig, topology: SystemTopologyModel) -> ResolvedToolchain:
        cmake = self._resolve_program("cmake")
        if cmake is None:
            raise RuntimeError("cmake is required but was not found in PATH")

        nvcc = self._resolve_program("nvcc")
        cuda_requested = config.enable_cuda != "off"
        if config.enable_cuda == "on" and nvcc is None:
            raise RuntimeError("build.enable_cuda=on but nvcc was not found in PATH")
        cuda_enabled = bool(cuda_requested and nvcc is not None and topology.gpus)

        candidates = self._compiler_candidates(config)
        if not candidates:
            raise RuntimeError("no C++ compiler was found; install GCC/Clang or set build.cxx_compiler")

        explicit = config.cuda_host_compiler or config.cxx_compiler
        diagnostics: list[str] = []
        for cxx in candidates:
            ok_cxx, cxx_output = self._probe_cxx20(cxx)
            if not ok_cxx:
                diagnostics.append(f"{cxx}: C++20 probe failed: {cxx_output}")
                if explicit:
                    break
                continue

            if cuda_enabled:
                assert nvcc is not None
                ok_cuda, cuda_output = self._probe_nvcc_host(nvcc, cxx)
                if not ok_cuda:
                    diagnostics.append(f"{cxx}: NVCC host-compiler probe failed: {cuda_output}")
                    if explicit:
                        break
                    continue
            return ResolvedToolchain(cmake=cmake, cxx=cxx, nvcc=nvcc, cuda_enabled=cuda_enabled)

        detail = "\n\n".join(diagnostics[-4:])
        if cuda_enabled:
            raise RuntimeError(
                "no single C++ compiler compatible with both C++20 and the detected NVCC was found. "
                "Install/select a CUDA-supported host compiler (for CUDA 12.4 this is typically GCC 13 or older) "
                "or set build.cxx_compiler/build.cuda_host_compiler to the same compiler.\n" + detail
            )
        raise RuntimeError("no usable C++20 compiler was found.\n" + detail)

    def build(self, config: BuildConfig, topology: SystemTopologyModel) -> BuildArtifact:
        toolchain = self.resolve_toolchain(config, topology)
        build_dir = (self.project_root / config.build_dir).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        # CMake >=3.24 is required by this project, and --fresh is available in
        # that version.  It removes stale CMakeCache/CMakeFiles while preserving
        # the build directory itself, preventing compiler/configuration drift.
        cmd = [
            toolchain.cmake,
            "--fresh",
            "-S",
            str(self.project_root),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={config.build_type}",
            f"-DCMAKE_CXX_COMPILER={toolchain.cxx}",
            f"-DPRBENCH_ENABLE_CUDA={'ON' if toolchain.cuda_enabled else 'OFF'}",
            f"-DPRBENCH_NATIVE_CPU_TUNING={'ON' if config.native_cpu_tuning else 'OFF'}",
        ]
        if toolchain.cuda_enabled:
            assert toolchain.nvcc is not None
            cmd += [
                f"-DCMAKE_CUDA_COMPILER={toolchain.nvcc}",
                f"-DCMAKE_CUDA_HOST_COMPILER={toolchain.cxx}",
            ]
            architectures = sorted(
                {
                    gpu.compute_capability.replace(".", "")
                    for gpu in topology.gpus
                    if gpu.compute_capability
                }
            )
            if architectures:
                cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(architectures)}")

        _run_checked(cmd)
        jobs = config.jobs or max(1, min(32, len(topology.allowed_cpus)))
        _run_checked([toolchain.cmake, "--build", str(build_dir), "--parallel", str(jobs)])
        worker = build_dir / "prbench-worker"
        if not worker.exists():
            raise RuntimeError(f"build completed but worker was not found at {worker}")

        import hashlib

        worker_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
        native_build_info: dict[str, object] = {}
        try:
            probe = subprocess.run(
                [str(worker), "--build-info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=15,
            )
            if probe.returncode == 0:
                for line in probe.stdout.splitlines():
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("event") == "build_info":
                        native_build_info = payload
                        break
        except (OSError, subprocess.SubprocessError):
            pass

        metadata = {
            "worker_sha256": worker_hash,
            "cmake_path": toolchain.cmake,
            "cmake_version": command_output([toolchain.cmake, "--version"]),
            "cxx_path": toolchain.cxx,
            "cxx_version": command_output([toolchain.cxx, "--version"]),
            "nvcc_path": toolchain.nvcc,
            "nvcc_version": command_output([toolchain.nvcc, "--version"]) if toolchain.nvcc else None,
            "cuda_host_compiler": toolchain.cxx if toolchain.cuda_enabled else None,
            "cuda_enabled": toolchain.cuda_enabled,
            "cuda_architectures": [g.compute_capability for g in topology.gpus],
            "build_type": config.build_type,
            "native_cpu_tuning": config.native_cpu_tuning,
            "fresh_cmake_configure": True,
            "native_build_info": native_build_info,
        }
        return BuildArtifact(worker.resolve(), toolchain.cuda_enabled, metadata)
