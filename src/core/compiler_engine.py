import os
import subprocess

class CompilerEngine:
    """
    Just-In-Time (JIT) Compiler Engine.
    Responsible for compiling C++/CUDA codes with specific macros (like DATA_TYPE)
    and hardware-specific optimizations (like -arch=sm_89).
    """
    def __init__(self, workspace_dir: str = "temp_workspace"):
        self.workspace_dir = workspace_dir
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def build_executable(self, task: dict, gpu_info: list) -> str:
        """
        Compiles the source code defined in the task configuration.
        Returns the absolute path to the generated binary executable.
        """
        exp_id = task.get("experiment_id", "UNKNOWN_EXP")
        algo_path = task.get("algorithm_path", "")
        data_type = task.get("data_type", "float32")
        flags_config = task.get("compiler_flags", "auto")

        # Map Python/YAML types to C++ standard types
        cpp_type_map = {
            "int32": "int32_t",
            "int64": "int64_t",
            "float32": "float",
            "float64": "double",
            "double": "double"
        }
        cpp_type = cpp_type_map.get(data_type, "float")

        # Define the output binary path
        out_filename = f"{exp_id}_{data_type}.out"
        out_filepath = os.path.join(self.workspace_dir, out_filename)

        # Skip compilation if the binary already exists
        if os.path.exists(out_filepath):
            print(f"[Compiler] Reusing existing binary: {out_filename}")
            return os.path.abspath(out_filepath)

        print(f"[Compiler] Building {exp_id} for data type: {data_type} ...")

        # Resolve the target source file (CUDA or standard C++)
        source_file = os.path.join(algo_path, "main.cu")
        if not os.path.exists(source_file):
            source_file = os.path.join(algo_path, "main.cpp")
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"Could not find main.cu or main.cpp in {algo_path}")

        # Determine compiler based on extension
        is_cuda = source_file.endswith(".cu")
        compiler = "nvcc" if is_cuda else "g++"

        # Construct the compilation command
        # -O3 ensures max optimization
        cmd = [compiler, source_file, "-o", out_filepath, "-O3", f"-D DATA_TYPE={cpp_type}"]

        # Hardware-specific auto-tuning for NVIDIA GPUs
        if is_cuda and flags_config == "auto" and gpu_info:
            # Use the compute capability of the first available GPU (e.g., 8.9 -> 89)
            cc = gpu_info[0].get("compute_capability", "8.9").replace(".", "")
            cmd.append(f"-arch=sm_{cc}")
        elif flags_config != "auto":
            # Append custom flags from the YAML configuration if provided
            cmd.extend(flags_config.split())

        # Execute the compilation process
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"[Compiler] Successfully built: {out_filepath}")
        except subprocess.CalledProcessError as e:
            print(f"[Compiler Error] Failed to compile {source_file}")
            print("--- COMPILER OUTPUT ---")
            print(e.stderr)
            print("-----------------------")
            raise RuntimeError("Compilation failed.")

        return os.path.abspath(out_filepath)


# Execution block for testing
if __name__ == "__main__":
    # Mock data to simulate the orchestrator passing a task
    mock_task_1 = {
        "experiment_id": "EXP_001_CUDA_ATOMICS",
        "algorithm_path": "algorithms/gpu_single/exp_001_cuda_atomics",
        "data_type": "float32",
        "compiler_flags": "auto"
    }
    
    mock_task_2 = {
        "experiment_id": "EXP_001_CUDA_ATOMICS",
        "algorithm_path": "algorithms/gpu_single/exp_001_cuda_atomics",
        "data_type": "double",
        "compiler_flags": "auto"
    }

    mock_gpu_info = [{"compute_capability": "8.9"}]

    engine = CompilerEngine()
    
    try:
        bin1 = engine.build_executable(mock_task_1, mock_gpu_info)
        bin2 = engine.build_executable(mock_task_2, mock_gpu_info)
        
        print("\nTest Execution:")
        subprocess.run([bin1])
        subprocess.run([bin2])
        
    except Exception as err:
        print(err)