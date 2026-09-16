#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "prbench/types.hpp"

namespace prbench {

struct DeviceMetrics {
    int device_id{-1};
    double h2d_us{0.0};
    double kernel_us{0.0};
    double d2h_us{0.0};
    double device_overhead_us{0.0};
    double total_us{0.0};
    double storage_read_us{0.0};
    double host_staging_us{0.0};
    std::uint64_t storage_read_bytes{0};
    std::uint64_t h2d_bytes{0};
    std::uint64_t d2h_bytes{0};
    std::uint64_t remote_host_read_bytes{0};
    std::size_t chunks{0};
    std::size_t elements{0};
    bool numa_requested{false};
    bool numa_applied{false};
};

struct CpuMetrics {
    double compute_us{0.0};
    double storage_read_us{0.0};
    std::uint64_t storage_read_bytes{0};
    std::size_t chunks{0};
    std::size_t elements{0};
};

struct CalibrationSample {
    std::string worker_kind;  // cpu or gpu
    int worker_index{0};
    int device_id{-1};
    std::size_t elements{0};
    double elapsed_us{0.0};
    double throughput_elements_s{0.0};
};

struct LinearModelMetrics {
    std::string worker_kind;
    int worker_index{0};
    int device_id{-1};
    double intercept_us{0.0};
    double slope_us_per_element{0.0};
};

struct PartitionMetrics {
    std::string worker_kind;
    int worker_index{0};
    int device_id{-1};
    std::size_t offset{0};
    std::size_t elements{0};
};

struct PrepareMetrics {
    std::vector<CalibrationSample> calibration_samples;
    std::vector<LinearModelMetrics> linear_models;
    std::vector<PartitionMetrics> partition;
    std::vector<double> initial_throughput_elements_s;
};

struct IterationMetrics {
    Value result;
    double e2e_us{0.0};
    double scheduler_us{0.0};
    double merge_us{0.0};
    CpuMetrics cpu;
    std::vector<DeviceMetrics> gpus;
    std::vector<double> worker_throughput_elements_s;
};

struct PartialResult {
    Value result;
    DeviceMetrics device;
};

}  // namespace prbench
