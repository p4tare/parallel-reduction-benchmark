import os
import sys
import time
import json
import subprocess
import psutil
import ast

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Fallback import logic for testing
try:
    from src.execution.energy_profilers import HardwareProfiler
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.execution.energy_profilers import HardwareProfiler


class TaskRunner:
    """
    Executes a single task (compiled C++ binary).
    Handles thermal throttling prevention, CPU affinity (taskset), 
    and merges execution time with energy profiling data.
    """
    def __init__(self, topology: dict, global_config: dict):
        self.topology = topology
        self.global_config = global_config
        
        self.gpu_temp_threshold = self.global_config.get("cooling_threshold_gpu_c", 50.0)
        self.cpu_temp_threshold = self.global_config.get("cooling_threshold_cpu_c", 65.0)

    def _get_max_cpu_temp(self) -> float:
        """Reads the maximum reported CPU temperature from psutil."""
        if not hasattr(psutil, "sensors_temperatures"):
            return 0.0
            
        temps = psutil.sensors_temperatures()
        if not temps:
            return 0.0
            
        max_temp = 0.0
        for name, entries in temps.items():
            for entry in entries:
                if entry.current > max_temp:
                    max_temp = entry.current
        return max_temp

    def _get_max_gpu_temp(self, gpu_allocation_str: str) -> float:
        """Reads the maximum temperature across all requested GPUs using NVML."""
        if not NVML_AVAILABLE:
            return 0.0
            
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            max_temp = 0.0
            
            # Safely parse the string "[0, 1]" into a list
            try:
                gpu_ids = ast.literal_eval(gpu_allocation_str)
                if not isinstance(gpu_ids, list):
                    gpu_ids = []
            except Exception:
                gpu_ids = []
            
            for i in gpu_ids:
                if i < device_count:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    if temp > max_temp:
                        max_temp = float(temp)
                        
            pynvml.nvmlShutdown()
            return max_temp
        except pynvml.NVMLError:
            return 0.0

    def _enforce_cooldown(self, gpu_allocation_str: str):
        """Blocks execution until hardware cools down below target thresholds."""
        print("[Runner] Checking thermal conditions...")
        
        while True:
            cpu_t = self._get_max_cpu_temp()
            gpu_t = self._get_max_gpu_temp(gpu_allocation_str)
            
            cpu_ok = cpu_t <= self.cpu_temp_threshold or cpu_t == 0.0
            gpu_ok = gpu_t <= self.gpu_temp_threshold or gpu_t == 0.0
            
            if cpu_ok and gpu_ok:
                break
                
            print(f"  -> Waiting for cooldown. Current: CPU {cpu_t}C, GPU {gpu_t}C. Target: CPU <= {self.cpu_temp_threshold}C, GPU <= {self.gpu_temp_threshold}C")
            time.sleep(2.0)

    def _build_cpu_affinity_mask(self, strategy: str) -> str:
        """Converts allocation strategy to a comma-separated list of core IDs for taskset."""
        p_cores = self.topology.get("p_cores", [])
        e_cores = self.topology.get("e_cores", [])
        physical_only = self.topology.get("physical_only", [])
        all_cores = list(range(self.topology.get("logical_cores", 1)))

        if strategy == "p_cores_only" and p_cores:
            selected = p_cores
        elif strategy == "e_cores_only" and e_cores:
            selected = e_cores
        elif strategy == "physical_cores_only" and physical_only:
            selected = physical_only
        else:
            selected = all_cores
            
        return ",".join(map(str, selected))

    def execute_task(self, task: dict, binary_path: str, dataset_path: str) -> dict:
        """
        Executes the binary within the profiler context, enforces limits, 
        and returns a unified dictionary of all metrics.
        """
        gpu_allocation_str = str(task.get("gpu_allocation", "[]"))
        
        try:
            target_gpu_ids = ast.literal_eval(gpu_allocation_str)
            if not isinstance(target_gpu_ids, list):
                target_gpu_ids = [0]
        except Exception:
            target_gpu_ids = [0]
            
        # Convert list [0, 1] to string "0,1" for the C++ wrapper arguments
        gpus_arg = ",".join(map(str, target_gpu_ids))
        
        # 1. Wait for hardware to cool down
        self._enforce_cooldown(gpu_allocation_str)
        
        # 2. Prepare execution arguments
        affinity_mask = self._build_cpu_affinity_mask(task.get("cpu_allocation_strategy", "all"))
        dedicated_threads_flag = "1" if task.get("use_dedicated_gpu_threads") else "0"
        
        cmd = [
            "taskset", "-c", affinity_mask,
            binary_path,
            "--data", dataset_path,
            "--gpus", gpus_arg,
            "--reps", str(task.get("repetitions", 10)),
            "--warmup", str(task.get("warmup_runs", 3)),
            "--dedicated-threads", dedicated_threads_flag,
            "--block-size", str(task.get("cuda_block_size", 256)),
            "--trace", "1"
        ]


        
        print(f"[Runner] Launching: {' '.join(cmd)}")
        
        # 3. Run with Profiler (passing the list of IDs instead of a single ID)
        polling_rate = self.global_config.get("polling_interval_ms", 10)
        
        with HardwareProfiler(polling_interval_ms=polling_rate, gpu_ids=target_gpu_ids) as profiler:
            try:
                # We expect the C++ code to print a valid JSON string to standard output.
                process = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    check=True
                )
                raw_output = process.stdout
            except subprocess.CalledProcessError as e:
                print(f"[Runner Error] Execution failed. STDERR:\n{e.stderr}")
                raw_output = "{}"
                
        # 4. Gather Energy metrics
        energy_metrics = profiler.get_results()
        
        # 5. Parse C++ JSON metrics
        algo_metrics = {}
        trace_lines = []
        
        for line in raw_output.splitlines():
            line = line.strip()
            # If line is JSON, parse it
            if line.startswith("{") and line.endswith("}"):
                try:
                    algo_metrics = json.loads(line)
                    continue # Skip adding JSON to the trace log
                except json.JSONDecodeError:
                    pass
            # Everything else is considered a trace log from C++ printf
            if line:
                trace_lines.append(line)

        # 6. Merge configurations and metrics
        final_result = {**task, **energy_metrics, **algo_metrics}
        final_result["trace_log"] = "\n".join(trace_lines) # Inject trace log into result dictionary
        
        return final_result

# Execution block for testing
if __name__ == "__main__":
    print("Testing Task Runner directly...")