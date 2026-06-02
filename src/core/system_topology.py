import platform
import psutil
import cpuinfo
import os

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class SystemTopology:
    """
    Class responsible for scanning and mapping hardware topology 
    (CPU, RAM, GPU). Acts as the Hardware Abstraction Layer for the Orchestrator.
    """
    
    def __init__(self):
        self.os_info = platform.system() + " " + platform.release()
        self.cpu_info = self._scan_cpu()
        self.gpu_info = self._scan_gpus()

    def _parse_siblings_count(self, raw_str: str) -> int:
        """Helper method to correctly count items in Linux sysfs lists (handles ranges)."""
        count = 0
        if not raw_str:
            return 1
            
        parts = raw_str.split(',')
        for part in parts:
            if '-' in part:
                try:
                    start, end = part.split('-')
                    count += (int(end) - int(start) + 1)
                except ValueError:
                    count += 1
            else:
                count += 1
        return count
        
    def _scan_cpu(self) -> dict:
        """Fetches information about the CPU, RAM, and maps P/E cores."""
        info = cpuinfo.get_cpu_info()
        
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        pe_mapping = self._detect_p_e_cores(logical_cores)
        
        return {
            "brand": info.get("brand_raw", "Unknown CPU"),
            "architecture": info.get("arch", "Unknown Arch"),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "hyperthreading_enabled": logical_cores > physical_cores,
            "total_ram_gb": ram_gb,
            "p_cores": pe_mapping["p_cores"],
            "e_cores": pe_mapping["e_cores"]
        }

    def _detect_p_e_cores(self, logical_count: int) -> dict:
        """
        Attempts to map logical core IDs to Performance (P) and Efficiency (E) cores.
        Utilizes Linux sysfs thread_siblings_list.
        """
        p_cores = []
        e_cores = []
        
        if platform.system() != "Linux":
            return {"p_cores": list(range(logical_count)), "e_cores": []}

        try:
            for i in range(logical_count):
                topology_path = f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
                
                if os.path.exists(topology_path):
                    with open(topology_path, "r") as f:
                        siblings_raw = f.read().strip()
                        thread_count = self._parse_siblings_count(siblings_raw)
                        
                        # P-Cores have Hyper-Threading (thread_count > 1)
                        # E-Cores do not have Hyper-Threading (thread_count == 1)
                        if thread_count > 1:
                            p_cores.append(i)
                        else:
                            e_cores.append(i)
            
            if p_cores or e_cores:
                return {
                    "p_cores": p_cores,
                    "e_cores": e_cores
                }
                
        except Exception as e:
            print(f"[Warning] Failed to map P/E cores via sysfs: {e}")
            
        return {"p_cores": list(range(logical_count)), "e_cores": []}

    def _scan_gpus(self) -> list:
        """Scans the system for NVIDIA GPUs using NVML."""
        gpus = []
        if not NVML_AVAILABLE:
            return gpus

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                    
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_gb = round(mem_info.total / (1024**3), 2)
                
                compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode('utf-8')
                    
                gpus.append({
                    "id": i,
                    "name": name,
                    "vram_gb": vram_gb,
                    "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
                    "uuid": uuid
                })
            pynvml.nvmlShutdown()
            
        except pynvml.NVMLError as e:
            print(f"[Warning] NVML encountered an error during scanning: {e}")
            
        return gpus

    def print_summary(self):
        """Prints a formatted summary of the detected hardware."""
        print("=" * 50)
        print("HARDWARE TOPOLOGY DETECTION")
        print("=" * 50)
        print(f"OS: {self.os_info}")
        print("\nCPU:")
        print(f"    - Model: {self.cpu_info['brand']}")
        print(f"    - Architecture: {self.cpu_info['architecture']}")
        print(f"    - Physical Cores: {self.cpu_info['physical_cores']}")
        print(f"    - Logical Cores: {self.cpu_info['logical_cores']}")
        print(f"    - Hyper-Threading: {'Yes' if self.cpu_info['hyperthreading_enabled'] else 'No'}")
        print(f"    - Total RAM: {self.cpu_info['total_ram_gb']} GB")
        
        print(f"\n    [Core Mapping]")
        print(f"    - Performance Cores (P-Cores) IDs: {self.cpu_info['p_cores']}")
        print(f"    - Efficiency Cores (E-Cores) IDs: {self.cpu_info['e_cores']}")
        
        print("\nGPU(s):")
        if not self.gpu_info:
            print("    [No NVIDIA GPUs detected / No drivers installed]")
        else:
            for gpu in self.gpu_info:
                print(f"    [{gpu['id']}] {gpu['name']} ({gpu['vram_gb']} GB VRAM)")
                print(f"        |- Compute Capability: {gpu['compute_capability']}")
                print(f"        |- UUID: {gpu['uuid']}")
        print("=" * 50)


if __name__ == "__main__":
    topology = SystemTopology()
    topology.print_summary()