import threading
import time
import os

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class HardwareProfiler:
    """
    Context Manager for profiling hardware energy consumption.
    It reads Intel RAPL counters for CPU energy and spawns a background 
    thread to poll NVML for GPU power usage (Watts) at a high frequency.
    """
    def __init__(self, polling_interval_ms: int = 10, gpu_id: int = 0):
        self.polling_interval = polling_interval_ms / 1000.0
        self.gpu_id = gpu_id
        
        self.running = False
        self.thread = None
        
        self.gpu_power_samples = []
        self.cpu_energy_start = 0
        self.cpu_energy_end = 0
        self.rapl_accessible = True

    def _read_rapl_uj(self) -> int:
        """
        Reads Intel RAPL energy counter in microjoules.
        Path targets the entire package (CPU socket).
        """
        path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return int(f.read().strip())
            except PermissionError:
                # On some strict Linux kernels, RAPL requires root/sudo
                self.rapl_accessible = False
                return 0
        self.rapl_accessible = False
        return 0

    def _poll_nvml(self):
        """Background thread function to poll GPU power."""
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                while self.running:
                    # nvmlDeviceGetPowerUsage returns milliwatts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    self.gpu_power_samples.append(power_mw / 1000.0)
                    time.sleep(self.polling_interval)
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                print(f"[Profiler Error] NVML polling failed: {e}")

    def __enter__(self):
        """Starts the profiling session."""
        self.gpu_power_samples = []
        self.cpu_energy_start = self._read_rapl_uj()
        
        self.running = True
        if NVML_AVAILABLE:
            self.thread = threading.Thread(target=self._poll_nvml)
            self.thread.start()
            
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the profiling session and aggregates data."""
        self.end_time = time.perf_counter()
        
        # Stop background thread
        self.running = False
        if self.thread:
            self.thread.join()
            
        self.cpu_energy_end = self._read_rapl_uj()

    def get_results(self) -> dict:
        """Calculates total energy in Joules."""
        duration_s = self.end_time - self.start_time
        
        # CPU Energy (Joules)
        if self.rapl_accessible:
            cpu_joules = (self.cpu_energy_end - self.cpu_energy_start) / 1_000_000.0
        else:
            cpu_joules = -1.0
            
        # GPU Energy (Joules) = Average Power (W) * Time (s)
        gpu_joules = 0.0
        avg_gpu_power_w = 0.0
        
        if self.gpu_power_samples:
            avg_gpu_power_w = sum(self.gpu_power_samples) / len(self.gpu_power_samples)
            gpu_joules = avg_gpu_power_w * duration_s
            
        return {
            "wall_time_s": round(duration_s, 4),
            "cpu_energy_j": round(cpu_joules, 4),
            "gpu_energy_j": round(gpu_joules, 4),
            "avg_gpu_power_w": round(avg_gpu_power_w, 2),
            "nvml_samples_count": len(self.gpu_power_samples),
            "rapl_permission_ok": self.rapl_accessible
        }

# Execution block for testing
if __name__ == "__main__":
    print("Testing Hardware Energy Profiler...")
    print("Simulating a 2-second heavy workload...")
    
    # Run the profiler context manager
    with HardwareProfiler(polling_interval_ms=10, gpu_id=0) as profiler:
        # Simulate CPU/GPU work
        # In the real system, this is where we will launch the C++ binary!
        start = time.time()
        while time.time() - start < 2.0:
            pass 
            
    results = profiler.get_results()
    
    print("\n[PROFILING RESULTS]")
    for key, value in results.items():
        print(f"  {key}: {value}")
        
    if not results["rapl_permission_ok"]:
        print("\n[NOTE] RAPL returned -1.0 Joules. This means your Linux kernel restricts")
        print("read access to CPU energy counters. To fix this during real experiments,")
        print("you might need to run the orchestrator with 'sudo' or adjust sysfs permissions.")