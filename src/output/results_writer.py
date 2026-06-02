import os
import csv
import json
from datetime import datetime

class ResultsWriter:
    """
    Handles the aggregation and disk writing of experiment results.
    Creates timestamped directories and saves both CSV metrics and JSON topology data.
    """
    def __init__(self, output_base_dir: str = "results"):
        self.output_base_dir = output_base_dir
        
        # Create a unique timestamped directory for this execution batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_base_dir, f"run_{timestamp}")
        
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def save_system_info(self, topology_data: dict, global_config: dict):
        """Saves the hardware map and global run settings as a JSON log."""
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
        """Writes the list of merged task results to a single CSV file."""
        if not results_list:
            print("[ResultsWriter] No results to save.")
            return

        path = os.path.join(self.run_dir, "measurements.csv")
        
        # Extract all unique keys dynamically (some algorithms might return custom metrics)
        fieldnames = set()
        for r in results_list:
            fieldnames.update(r.keys())
        
        # Sort fields to put the most important ones first
        priority_cols = [
            "experiment_id", "data_size", "data_type", "cuda_block_size", 
            "use_dedicated_gpu_threads", "wall_time_s", "cpu_energy_j", "gpu_energy_j"
        ]
        sorted_fields = [f for f in priority_cols if f in fieldnames]
        sorted_fields += sorted([f for f in fieldnames if f not in sorted_fields])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fields)
            writer.writeheader()
            for row in results_list:
                writer.writerow(row)
                
        print(f"[ResultsWriter] Saved {len(results_list)} rows to {path}")