#include "prbench/integrated_strategy.hpp"

#include <stdexcept>

namespace prbench {

bool uses_integrated_strategy(const WorkerConfig& config) noexcept {
    return config.storage_policy != "host_resident" || config.memory_path != "default";
}

std::string integrated_unsupported_reason(const WorkerConfig& config) {
    if (config.storage_policy != "host_resident") {
        return "integrated file/GPU memory paths require a CUDA-enabled build in this version";
    }
    if (config.memory_path != "default") {
        return "requested GPU memory path requires a CUDA-enabled build";
    }
    return {};
}

std::unique_ptr<IReductionStrategy> make_integrated_strategy(const WorkerConfig&) {
    throw std::runtime_error("integrated CUDA strategy requested in CPU-only build");
}

}  // namespace prbench
