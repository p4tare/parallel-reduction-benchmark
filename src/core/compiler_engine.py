import os
import subprocess

class CompilerEngine:
    """
    Smart Just-In-Time (JIT) Compiler Engine.
    Compiles C++/CUDA codes using a universal wrapper template.
    Uses timestamp-based caching to automatically rebuild if source code changes.
    """
    def __init__(self):
        pass

    def build_executable(self, task: dict, gpu_info: list) -> str:
        """
        Compiles the source code defined in the task configuration.
        Places the binary inside the algorithm's directory.
        Returns the absolute path to the generated binary executable.
        """
        algo_path = task.get("algorithm_path", "")
        data_type = task.get("data_type", "float32")
        flags_config = task.get("compiler_flags", "auto")

        if not os.path.exists(algo_path):
            raise FileNotFoundError(f"Algorithm path does not exist: {algo_path}")

        # STRIP PRECISION SUFFIX (e.g., float32_3 -> float32)
        base_data_type = data_type.split("_")[0]

        cpp_type_map = {
            "int32": "int32_t",
            "int64": "int64_t",
            "float32": "float",
            "float64": "double",
            "double": "double"
        }
        
        # Use the stripped base type for mapping
        cpp_type = cpp_type_map.get(base_data_type, "float")

        wrapper_file = os.path.join("algorithms", "cuda_wrapper_template.cu")
        if not os.path.exists(wrapper_file):
            raise FileNotFoundError(f"Could not find universal wrapper: {wrapper_file}")
            
        kernel_file = os.path.join(algo_path, "kernel.cuh")
        if not os.path.exists(kernel_file):
            raise FileNotFoundError(f"Could not find algorithm logic: {kernel_file}")

        # Use the stripped base type for the binary name to reuse it across different precisions
        out_filename = f"compiled_bin_{base_data_type}.out"
        out_filepath = os.path.join(algo_path, out_filename)

        if os.path.exists(out_filepath):
            bin_mtime = os.path.getmtime(out_filepath)
            wrapper_mtime = os.path.getmtime(wrapper_file)
            kernel_mtime = os.path.getmtime(kernel_file)
            
            if bin_mtime > wrapper_mtime and bin_mtime > kernel_mtime:
                print(f"[Compiler] Reusing existing binary (up to date): {out_filename}")
                return os.path.abspath(out_filepath)
            else:
                print(f"[Compiler] Source code changed. Rebuilding {out_filename}...")

        print(f"[Compiler] Building {algo_path} for type: {base_data_type} ...")

        compiler = "nvcc"
        
        cmd = [
            compiler, wrapper_file, 
            "-o", out_filepath, 
            "-O3", 
            f"-DDATA_TYPE={cpp_type}", 
            "-I", algo_path,
            "-Xcompiler", "-fopenmp"
        ]

        if flags_config == "auto" and gpu_info:
            cc = gpu_info[0].get("compute_capability", "8.9").replace(".", "")
            cmd.append(f"-arch=sm_{cc}")
        elif flags_config != "auto":
            cmd.extend(flags_config.split())

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"[Compiler] Successfully built: {out_filepath}")
        except subprocess.CalledProcessError as e:
            print(f"[Compiler Error] Failed to compile algorithm in {algo_path}")
            print("--- COMPILER OUTPUT ---")
            print(e.stderr)
            print("-----------------------")
            raise RuntimeError("Compilation failed.")

        return os.path.abspath(out_filepath)

if __name__ == "__main__":
    print("Compilation module ready.")