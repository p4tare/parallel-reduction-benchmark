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
        Expands the experiment configurations into a flat list of individual tasks.
        Each task represents a single run with a specific combination of parameters.
        """
        tasks = []
        
        for exp in self.experiments:
            # Base parameters that do not change during this experiment
            base_task = {
                "experiment_id": exp.get("id", "UNKNOWN_EXP"),
                "algorithm_path": exp.get("algorithm_path", ""),
                "compiler_flags": exp.get("compiler_flags", "auto"),
                "repetitions": exp.get("repetitions", 10),
                "warmup_runs": exp.get("warmup_runs", 3),
                "data_generation_mode": exp.get("data", {}).get("generation_mode", "random_uniform"),
                "gpu_allocation": exp.get("hardware", {}).get("gpu_allocation", "all"),
                "cpu_allocation_strategy": exp.get("hardware", {}).get("cpu_allocation_strategy", "all")
            }

            # Extract lists of parameters to form the grid
            # If a user provides a single value or omits it, we wrap it in a list with a default
            sizes = exp.get("data", {}).get("sizes", [1000])
            types = exp.get("data", {}).get("types", ["float32"])
            dedicated_threads = exp.get("hardware", {}).get("use_dedicated_gpu_threads", [False])
            block_sizes = exp.get("hardware", {}).get("cuda_block_sizes", [256])

            # Generate Cartesian product (all possible combinations)
            combinations = itertools.product(sizes, types, dedicated_threads, block_sizes)
            
            for combo in combinations:
                task = base_task.copy()
                task["data_size"] = combo[0]
                task["data_type"] = combo[1]
                task["use_dedicated_gpu_threads"] = combo[2]
                task["cuda_block_size"] = combo[3]
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
                print("-" * 40)
        print("=" * 60)

# Execution block for testing
if __name__ == "__main__":
    test_config_path = "configs/main_experiments.yaml"
    parser = ConfigParser(test_config_path)
    parser.print_summary()