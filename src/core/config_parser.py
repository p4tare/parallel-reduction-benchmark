import yaml
import os
import itertools

class ConfigParser:
    """
    Parses the YAML configuration file and generates a flat grid of tasks
    (Cartesian product of all array parameters) to be executed by the Runner.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.raw_config = self._load_yaml()
        self.global_settings = self.raw_config.get("global", {})
        self.experiments = self.raw_config.get("experiments", [])

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

    def generate_task_grid(self) -> list:
        """
        Parses all experiments and generates a flat list of tasks based on the Cartesian product
        of requested hardware and data parameters.
        """
        tasks = []
        # FIX: We now correctly iterate over self.experiments loaded in __init__
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

            gpu_allocs = hw_cfg.get("gpu_allocation", ["0"])
            gpu_allocs = gpu_allocs if isinstance(gpu_allocs, list) else [gpu_allocs]
            
            cpu_allocs = hw_cfg.get("cpu_allocation_strategy", ["all"])
            ded_threads = hw_cfg.get("use_dedicated_gpu_threads", [False])
            block_sizes = hw_cfg.get("cuda_block_sizes", [256])

            for size in sizes:
                for dtype in types:
                    for mode in modes: 
                        for gpu_alloc in gpu_allocs:
                            for cpu_alloc in cpu_allocs:
                                for ded_thread in ded_threads:
                                    for b_size in block_sizes:
                                        task = base_task.copy()
                                        task["data_size"] = size
                                        task["data_type"] = dtype
                                        task["data_generation_mode"] = mode
                                        task["gpu_allocation"] = gpu_alloc
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
            
        print(f"\nEXPERIMENTS LOADED: {len(self.experiments)}")
        print(f"TOTAL UNIQUE TASKS GENERATED: {len(tasks)}")
        
        if len(tasks) > 0:
            print("\nPREVIEW OF TASK QUEUE (First 3 tasks):")
            for i, task in enumerate(tasks[:3]):
                print(f"  Task {i+1}:")
                print(f"    ID: {task['experiment_id']}")
                print(f"    Size: {task['data_size']} | Type: {task['data_type']}")
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