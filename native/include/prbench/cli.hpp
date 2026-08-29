#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "prbench/types.hpp"

namespace prbench {

struct WorkerConfig {
    std::filesystem::path dataset_path;
    DataType dtype{DataType::Float32};
    ReductionOperation operation{ReductionOperation::Sum};
    std::size_t count{0};
    SchedulerKind scheduler{SchedulerKind::CpuOnly};
    CpuBackendKind cpu_backend{CpuBackendKind::Sequential};
    GpuBackendKind gpu_backend{GpuBackendKind::None};
    TransferPolicy transfer_policy{TransferPolicy::Sync};
    std::vector<int> gpu_ids;
    std::vector<int> cpu_affinity;
    std::vector<int> gpu_worker_cpus;
    int cpu_threads{1};
    int warmup_runs{5};
    int calibration_repetitions{5};
    std::size_t cache_rotation_target_bytes{268435456};
    std::size_t cache_rotation_max_replicas{64};
    int block_size{256};
    std::size_t chunk_size{1u << 20};
    std::size_t min_chunk_size{1u << 16};
    std::size_t max_chunk_size{1u << 24};
    double guided_factor{2.0};
    double target_chunk_ms{5.0};
    double ema_alpha{0.25};
    int pipeline_streams{4};
    int pipeline_chunks{16};
    std::size_t pipeline_chunk_elements{0};
    bool self_test{false};
    bool build_info{false};
    bool probe_cpu_types{false};
    std::vector<int> probe_cpus;
};

WorkerConfig parse_cli(int argc, char** argv);

}  // namespace prbench
