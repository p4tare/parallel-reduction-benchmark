import threading
import time
import os
import glob

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class HardwareProfiler:
    """
    Context Manager for profiling hardware energy consumption.
    Reads Intel RAPL counters for ALL CPU sockets (Multi-CPU aware) 
    and spawns a background thread to poll NVML for power usage across MULTIPLE GPUs.
    """
    def __init__(self, polling_interval_ms: int = 10, gpu_ids: list = None):
        self.polling_interval = polling_interval_ms / 1000.0
        self.gpu_ids = gpu_ids if gpu_ids is not None else [0]
        
        self.running = False
        self.thread = None
        
        # Dictionary to store power samples independently for each requested GPU
        self.gpu_power_samples = {g_id: [] for g_id in self.gpu_ids}
        
        self.cpu_energy_start = 0
        self.cpu_energy_end = 0
        self.rapl_accessible = True

    def _read_rapl_uj_total(self) -> int:
        """
        Reads Intel RAPL energy counter in microjoules.
        Dynamically detects all CPU packages (intel-rapl:0, intel-rapl:1) for NUMA systems
        and sums their energy consumption.
        """
        rapl_base = "/sys/class/powercap/intel-rapl"
        if not os.path.exists(rapl_base):
            self.rapl_accessible = False
            return 0

        total_uj = 0
        try:
            # Match top-level domains like intel-rapl:0, intel-rapl:1 (ignoring subdomains)
            packages = [d for d in os.listdir(rapl_base) if ":" in d and d.count(":") == 1]
            if not packages:
                self.rapl_accessible = False
                return 0
                
            for pkg in packages:
                path = os.path.join(rapl_base, pkg, "energy_uj")
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        total_uj += int(f.read().strip())
                        
            return total_uj
        except PermissionError:
            self.rapl_accessible = False
            return 0

    def _poll_nvml(self):
        """Background thread function to poll GPU power for all specified GPUs."""
        if NVML_AVAILABLE and self.gpu_ids:
            try:
                pynvml.nvmlInit()
                # Create a dictionary of handles for fast access
                handles = {}
                for g_id in self.gpu_ids:
                    try:
                        handles[g_id] = pynvml.nvmlDeviceGetHandleByIndex(int(g_id))
                    except pynvml.NVMLError:
                        print(f"[Profiler Warning] GPU {g_id} not found by NVML.")
                
                while self.running:
                    for g_id, handle in handles.items():
                        try:
                            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                            self.gpu_power_samples[g_id].append(power_mw / 1000.0)
                        except pynvml.NVMLError:
                            pass
                    time.sleep(self.polling_interval)
                    
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                print(f"[Profiler Error] NVML polling failed: {e}")

    def __enter__(self):
        """Starts the profiling session."""
        for g_id in self.gpu_ids:
            self.gpu_power_samples[g_id] = []
            
        self.cpu_energy_start = self._read_rapl_uj_total()
        
        self.running = True
        if NVML_AVAILABLE:
            self.thread = threading.Thread(target=self._poll_nvml)
            self.thread.start()
            
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the profiling session and aggregates data."""
        self.end_time = time.perf_counter()
        
        self.running = False
        if self.thread:
            self.thread.join()
            
        self.cpu_energy_end = self._read_rapl_uj_total()

    def get_results(self) -> dict:
        """Calculates total energy in Joules for all components."""
        duration_s = self.end_time - self.start_time
        
        if self.rapl_accessible:
            cpu_joules = (self.cpu_energy_end - self.cpu_energy_start) / 1_000_000.0
        else:
            cpu_joules = -1.0
            
        results = {
            "wall_time_s": round(duration_s, 4),
            "cpu_energy_j": round(cpu_joules, 4),
            "rapl_permission_ok": self.rapl_accessible
        }
        
        total_gpu_joules = 0.0
        max_samples = 0
        total_avg_power_w = 0.0
        
        # Calculate dynamic metrics for each requested GPU
        for g_id in self.gpu_ids:
            samples = self.gpu_power_samples.get(g_id, [])
            sample_count = len(samples)
            if sample_count > max_samples:
                max_samples = sample_count
                
            avg_power_w = sum(samples) / sample_count if sample_count > 0 else 0.0
            gpu_j = avg_power_w * duration_s
            
            # Dynamic columns for detailed Multi-GPU tracking
            results[f"gpu_{g_id}_power_w"] = round(avg_power_w, 2)
            results[f"gpu_{g_id}_energy_j"] = round(gpu_j, 4)
            
            total_avg_power_w += avg_power_w
            total_gpu_joules += gpu_j
            
        # Global aggregated GPU metrics (for simple Single-GPU tests and backwards compatibility)
        results["gpu_energy_j"] = round(total_gpu_joules, 4)
        results["avg_gpu_power_w"] = round(total_avg_power_w, 2)
        results["nvml_samples_count"] = max_samples
            
        return results

if __name__ == "__main__":
    print("Testing Multi-GPU Hardware Energy Profiler...")
    print("Simulating a 2-second heavy workload on GPUs [0, 1]...")
    
    # Simulating request for two GPUs (will fallback gracefully if only 1 exists)
    with HardwareProfiler(polling_interval_ms=10, gpu_ids=[0, 1]) as profiler:
        start = time.time()
        while time.time() - start < 2.0:
            pass 
            
    results = profiler.get_results()
    
    print("\n[PROFILING RESULTS]")
    for key, value in results.items():
        print(f"  {key}: {value}")