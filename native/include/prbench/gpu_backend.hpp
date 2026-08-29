#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "prbench/metrics.hpp"
#include "prbench/types.hpp"

namespace prbench {

struct GpuReducerConfig {
    int device_id{0};
    GpuBackendKind backend{GpuBackendKind::Cub};
    TransferPolicy transfer_policy{TransferPolicy::Sync};
    DataType dtype{DataType::Float32};
    ReductionOperation operation{ReductionOperation::Sum};
    std::size_t max_elements{0};
    int block_size{256};
    int pipeline_streams{4};
    int pipeline_chunks{16};
};

class IGpuReducer {
public:
    virtual ~IGpuReducer() = default;
    virtual PartialResult reduce(const void* host_data, std::size_t count) = 0;
};

bool gpu_runtime_available() noexcept;
std::string gpu_unavailable_reason();
std::string gpu_library_versions_json();
std::unique_ptr<IGpuReducer> make_gpu_reducer(const GpuReducerConfig& config);

}  // namespace prbench
