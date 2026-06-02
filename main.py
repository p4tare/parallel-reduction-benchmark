import argparse
import os
import sys

from src.core.system_topology import SystemTopology
from src.core.config_parser import ConfigParser
from src.core.compiler_engine import CompilerEngine
from src.data.data_factory import DataFactory
from src.execution.runner import TaskRunner
from src.output.results_writer import ResultsWriter

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Hybrid Reduction Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    print("=" * 60)
    print("STARTING HYBRID REDUCTION BENCHMARK SYSTEM")
    print("=" * 60)

    # 2. Initialize Hardware Topology
    topology = SystemTopology()
    topology.print_summary()

    # 3. Parse Configuration
    try:
        config = ConfigParser(args.config)
    except FileNotFoundError:
        print(f"[Error] Could not find config file: {args.config}")
        sys.exit(1)
        
    config.print_summary()
    
    global_settings = config.get_global_settings()
    tasks = config.generate_task_grid()

    if not tasks:
        print("[Main] No tasks generated. Exiting.")
        sys.exit(0)

    # 4. Initialize Subsystems
    data_factory = DataFactory()
    compiler = CompilerEngine()
    
    # TaskRunner needs CPU topology specifically for taskset masks
    runner = TaskRunner(topology=topology.cpu_info, global_config=global_settings)
    writer = ResultsWriter(output_base_dir=global_settings.get("output_dir", "results"))

    # Save pre-run hardware state
    full_topology_dump = {
        "os": topology.os_info,
        "cpu": topology.cpu_info,
        "gpu": topology.gpu_info
    }
    writer.save_system_info(topology_data=full_topology_dump, global_config=global_settings)

    # 5. Main Execution Loop
    all_results = []
    total_tasks = len(tasks)

    for idx, task in enumerate(tasks):
        print(f"\n>>> [{idx+1}/{total_tasks}] Executing Task: {task['experiment_id']} "
              f"| Type: {task['data_type']} | Size: {task['data_size']}")
        
        # Step A: Generate / Stage Data
        try:
            data_path = data_factory.generate_data(
                size=task["data_size"],
                dtype_str=task["data_type"],
                mode=task["data_generation_mode"]
            )
        except Exception as e:
            print(f"  [Error] Data generation failed: {e}")
            continue

        # Step B: Compile Binary (JIT)
        try:
            binary_path = compiler.build_executable(task, topology.gpu_info)
        except Exception as e:
            print(f"  [Error] Compilation failed: {e}")
            continue

        # Step C: Run and Profile
        result_dict = runner.execute_task(task, binary_path, data_path)
        all_results.append(result_dict)

    # 6. Save Final Results
    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED. SAVING RESULTS...")
    print("=" * 60)
    writer.save_results_csv(all_results)
    print("[Main] Execution finished successfully.")

if __name__ == "__main__":
    main()