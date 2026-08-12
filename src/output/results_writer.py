import os
import csv
import json
import re
from datetime import datetime

class ResultsWriter:
    def __init__(self, output_base_dir: str = "results"):
        self.output_base_dir = output_base_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_base_dir, f"run_{timestamp}")
        if not os.path.exists(self.run_dir): os.makedirs(self.run_dir)

    def save_system_info(self, topology_data: dict, global_config: dict):
        info = {
            "hardware_topology": topology_data,
            "global_configuration": global_config,
            "timestamp_start": datetime.now().isoformat()
        }
        path = os.path.join(self.run_dir, "system_info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
        print(f"[ResultsWriter] Saved system info to {path}")

    def save_results_csv(self, results_list: list):
        if not results_list: return

        path = os.path.join(self.run_dir, "measurements.csv")

        def _flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
            items = []
            for k, v in d.items():
                if k in ["trace_log", "rapl_permission_ok"]: continue
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict): items.extend(_flatten_dict(v, new_key, sep=sep).items())
                else: items.append((new_key, v))
            return dict(items)

        flattened_results = [_flatten_dict(r) for r in results_list]
        all_keys = set(k for fr in flattened_results for k in fr.keys())

        # Zaktualizowana standardowa ramka
        preferred_order = [
            "experiment_id", "iteration", "data_size", "data_type", "data_generation_mode", 
            "cuda_block_size", "use_dedicated_gpu_threads", "gpu_allocation", 
            "wall_time_s", "cpu_energy_j", "gpu_energy_j", "avg_gpu_power_w", 
            "gpu_0_energy_j", "gpu_0_power_w", "gpu_1_energy_j", "gpu_1_power_w", 
            "algorithm_path", "compiler_flags", "cpp_cpu_time_us", 
            "cpu_allocation_strategy", "gpu_kernel_time_us", "is_correct", 
            "nvml_samples_count", "openmp_max_threads", "expected_result", 
            "reduction_result", "repetitions", "warmup_runs"
        ]

        fieldnames = [col for col in preferred_order if col in all_keys]
        for col in preferred_order:
            if col in all_keys: all_keys.remove(col)
        fieldnames.extend(sorted(list(all_keys)))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fr in flattened_results:
                writer.writerow(fr)
                
        print(f"[ResultsWriter] Saved {len(results_list)} repetition rows to {path}")

    def save_execution_traces(self, results_list: list):
        if not results_list: return
        path = os.path.join(self.run_dir, "execution_traces.log")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("=========================================================\n")
            f.write(" ALGORITHM EXECUTION TRACES (TELEMETRY LOGS)\n")
            f.write("=========================================================\n\n")
            
            seen_tasks = set()
            for r in results_list:
                task_sig = f"{r.get('experiment_id')}_{r.get('gpu_allocation')}_{r.get('cuda_block_size')}"
                if task_sig not in seen_tasks:
                    seen_tasks.add(task_sig)
                    trace = r.get("trace_log", "")
                    if trace:
                        f.write(f"--- Task ID: {r.get('experiment_id')} | GPUs: {r.get('gpu_allocation')} | Block: {r.get('cuda_block_size')} ---\n")
                        f.write(trace + "\n\n")
                    
        print(f"[ResultsWriter] Saved deduplicated telemetry logs to {path}")