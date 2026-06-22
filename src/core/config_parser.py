import yaml
import os
import ast
import random
import itertools
from src.core.system_topology import SystemTopology

class ConfigParser:
    """
    Parses the YAML configuration file and generates a flat grid of tasks.
    Features Smart GPU Allocation resolving ('all', 'random', '[0, 1]').
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.raw_config = self._load_yaml()
        self.global_settings = self.raw_config.get("global", {})
        self.experiments = self.raw_config.get("experiments", [])
        
        # FIXED: Use the correct class name SystemTopology and access gpu_info list
        topology_detector = SystemTopology()
        self.physical_gpu_count = len(topology_detector.gpu_info)

    def _load_yaml(self) -> dict:
        """Loads and parses the YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing YAML file: {exc}")

    def get_global_settings(self) -> dict:
        """Returns the global application settings."""
        return self.global_settings

    def _resolve_gpu_allocation(self, alloc_raw) -> str:
        """
        Translates human-readable GPU policies ('all', 'random', '[0,1]', '0') 
        into a string representing a list of physical GPU IDs.
        """
        alloc_str = str(alloc_raw).strip().lower()
        
        if self.physical_gpu_count == 0:
            return "[]"

        if alloc_str == "all":
            return str(list(range(self.physical_gpu_count)))
            
        if alloc_str == "random":
            return str([random.choice(range(self.physical_gpu_count))])
            
        if alloc_str.startswith("[") and alloc_str.endswith("]"):
            try:
                parsed_list = ast.literal_eval(alloc_str)
                if isinstance(parsed_list, list):
                    valid_gpus = [int(g) for g in parsed_list if int(g) < self.physical_gpu_count]
                    return str(valid_gpus)
            except Exception:
                pass 

        try:
            target = int(alloc_str)
            if target < self.physical_gpu_count:
                return str([target])
            else:
                return str([0])
        except ValueError:
            return str([0])

    def generate_task_grid(self) -> list:
        """
        Parses all experiments and generates a flat list of tasks based on the Cartesian product
        of requested hardware and data parameters.
        """
        tasks = []
        for exp in self.experiments:
            base_task = {
                "experiment_id": exp.get("id", "UNKNOWN"),
                "algorithm_path": exp.get("algorithm_path", ""),
                "compiler_flags": exp.get("compiler_flags", "auto"),
                "repetitions": exp.get("repetitions", 10),
                "warmup_runs": exp.get("warmup_runs", 3)
            }

            data_cfg = exp.get("data", {})
            hw_cfg = exp.get("hardware", {})

            sizes = data_cfg.get("sizes", [1000000])
            types = data_cfg.get("types", ["float32"])
            
            raw_modes = data_cfg.get("generation_mode", ["random_uniform"])
            modes = raw_modes if isinstance(raw_modes, list) else [raw_modes]

            raw_gpu_allocs = hw_cfg.get("gpu_allocation", ["all"])
            gpu_allocs_list = raw_gpu_allocs if isinstance(raw_gpu_allocs, list) else [raw_gpu_allocs]
            
            resolved_gpu_allocs = []
            for alloc in gpu_allocs_list:
                resolved_gpu_allocs.append(self._resolve_gpu_allocation(alloc))
            
            resolved_gpu_allocs = list(set(resolved_gpu_allocs))

            cpu_allocs = hw_cfg.get("cpu_allocation_strategy", ["all"])
            ded_threads = hw_cfg.get("use_dedicated_gpu_threads", [False])
            block_sizes = hw_cfg.get("cuda_block_sizes", [256])

            for size in sizes:
                for dtype in types:
                    for mode in modes: 
                        for gpu_alloc_str in resolved_gpu_allocs:
                            for cpu_alloc in cpu_allocs:
                                for ded_thread in ded_threads:
                                    for b_size in block_sizes:
                                        task = base_task.copy()
                                        task["data_size"] = size
                                        task["data_type"] = dtype
                                        task["data_generation_mode"] = mode
                                        task["gpu_allocation"] = gpu_alloc_str
                                        task["cpu_allocation_strategy"] = cpu_alloc
                                        task["use_dedicated_gpu_threads"] = ded_thread
                                        task["cuda_block_size"] = b_size
                                        tasks.append(task)
        return tasks

    def print_summary(self):
        """Prints a summary of the loaded configuration and generated tasks."""
        tasks = self.generate_task_grid()
        
        print("=" * 60)
        print("CONFIGURATION PARSER SUMMARY")
        print("=" * 60)
        print("GLOBAL SETTINGS:")
        for key, value in self.global_settings.items():
            print(f"    - {key}: {value}")
            
        print(f"\nDETECTED PHYSICAL GPUs: {self.physical_gpu_count}")
        print(f"EXPERIMENTS LOADED: {len(self.experiments)}")
        print(f"TOTAL UNIQUE TASKS GENERATED: {len(tasks)}")
        
        if len(tasks) > 0:
            print("\nPREVIEW OF TASK QUEUE (First 3 tasks):")
            for i, task in enumerate(tasks[:3]):
                print(f"  Task {i+1}:")
                print(f"    ID: {task['experiment_id']}")
                print(f"    Size: {task['data_size']} | Type: {task['data_type']}")
                print(f"    GPU Allocation (Resolved): {task['gpu_allocation']}")
                print(f"    Dedicated GPU Thread: {task['use_dedicated_gpu_threads']}")
                print(f"    CUDA Block Size: {task['cuda_block_size']}")
                print(f"    CPU Strategy: {task['cpu_allocation_strategy']}")
                print("-" * 40)
        print("=" * 60)

if __name__ == "__main__":
    test_config_path = "configs/main_experiments.yaml"
    if os.path.exists(test_config_path):
        parser = ConfigParser(test_config_path)
        parser.print_summary()
    else:
        print("Run from root directory.")