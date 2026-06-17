import os
import platform
import psutil
import cpuinfo

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class SystemTopology:
    """
    Detects hardware topology (CPU cores, RAM, GPUs).
    Maps CPU cores to Performance/Efficiency sets based on sysfs for taskset pinning.
    """
    def __init__(self):
        self.os_info = self._scan_os()
        self.cpu_info = self._scan_cpu()
        self.gpu_info = self._scan_gpus()

    def _scan_os(self) -> str:
        return f"{platform.system()} {platform.release()}"

    def _parse_siblings_count(self, siblings_raw: str) -> int:
        """Parses thread_siblings_list output like '0-1' or '0,1' into a count."""
        count = 0
        ranges = siblings_raw.split(',')
        for r in ranges:
            if '-' in r:
                start, end = map(int, r.split('-'))
                count += (end - start + 1)
            else:
                count += 1
        return count

    def _detect_core_topology(self, logical_count: int) -> dict:
        """
        Maps logical core IDs to Performance (P) and Efficiency (E) cores.
        Also isolates a list of strictly physical cores (ignoring Hyper-Threading siblings).
        """
        p_cores = []
        e_cores = []
        physical_only = []
        processed_physical_cores = set()
        
        if platform.system() != "Linux":
            return {"p_cores": list(range(logical_count)), "e_cores": [], "physical_only": list(range(logical_count))}

        try:
            for i in range(logical_count):
                # 1. P/E Detection via thread siblings count
                topology_path = f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
                
                if os.path.exists(topology_path):
                    with open(topology_path, "r") as f:
                        siblings_raw = f.read().strip()
                        thread_count = self._parse_siblings_count(siblings_raw)
                        
                        if thread_count > 1:
                            p_cores.append(i)
                        else:
                            e_cores.append(i)
                
                # 2. Physical Core Isolation
                core_id_path = f"/sys/devices/system/cpu/cpu{i}/topology/core_id"
                if os.path.exists(core_id_path):
                    with open(core_id_path, "r") as f:
                        core_id = int(f.read().strip())
                        # Add to physical_only if we haven't seen this physical core yet
                        if core_id not in processed_physical_cores:
                            physical_only.append(i)
                            processed_physical_cores.add(core_id)
            
            if p_cores or e_cores:
                return {
                    "p_cores": p_cores,
                    "e_cores": e_cores,
                    "physical_only": sorted(physical_only)
                }
                
        except Exception as e:
            print(f"[Warning] Failed to map topology via sysfs: {e}")
            
        return {"p_cores": list(range(logical_count)), "e_cores": [], "physical_only": list(range(logical_count))}

    def _scan_cpu(self) -> dict:
        """Fetches information about the CPU, RAM, and maps P/E cores."""
        info = cpuinfo.get_cpu_info()
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        topology_map = self._detect_core_topology(logical_cores)
        
        return {
            "brand": info.get("brand_raw", "Unknown CPU"),
            "architecture": info.get("arch", "Unknown Arch"),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "hyperthreading_enabled": logical_cores > physical_cores,
            "total_ram_gb": ram_gb,
            "p_cores": topology_map["p_cores"],
            "e_cores": topology_map["e_cores"],
            "physical_only": topology_map["physical_only"]
        }

    def _scan_gpus(self) -> list:
        """Scans for NVIDIA GPUs using NVML."""
        gpus = []
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    
                    gpus.append({
                        "id": i,
                        "name": name,
                        "vram_gb": round(mem_info.total / (1024**3), 2),
                        "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
                        "uuid": uuid
                    })
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                print(f"[Warning] NVML Error during GPU scan: {e}")
        return gpus

    def print_summary(self):
        """Prints the hardware topology in a readable format."""
        print("=" * 50)
        print("HARDWARE TOPOLOGY DETECTION")
        print("=" * 50)
        print(f"OS: {self.os_info}\n")
        
        print("CPU:")
        print(f"    - Model: {self.cpu_info['brand']}")
        print(f"    - Architecture: {self.cpu_info['architecture']}")
        print(f"    - Physical Cores: {self.cpu_info['physical_cores']}")
        print(f"    - Logical Cores: {self.cpu_info['logical_cores']}")
        print(f"    - Hyper-Threading: {'Yes' if self.cpu_info['hyperthreading_enabled'] else 'No'}")
        print(f"    - Total RAM: {self.cpu_info['total_ram_gb']} GB\n")
        
        print("    [Core Mapping]")
        print(f"    - Performance Cores (P-Cores) IDs: {self.cpu_info['p_cores']}")
        print(f"    - Efficiency Cores (E-Cores) IDs: {self.cpu_info['e_cores']}")
        print(f"    - Strictly Physical Cores IDs: {self.cpu_info['physical_only']}\n")
        
        print("GPU(s):")
        if not self.gpu_info:
            print("    No NVIDIA GPUs detected.")
        for gpu in self.gpu_info:
            print(f"    [{gpu['id']}] {gpu['name']} ({gpu['vram_gb']} GB VRAM)")
            print(f"        |- Compute Capability: {gpu['compute_capability']}")
            print(f"        |- UUID: {gpu['uuid']}")
        print("=" * 50)

if __name__ == "__main__":
    topology = SystemTopology()
    topology.print_summary()