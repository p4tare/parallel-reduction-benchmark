#include "prbench/gpu_backend.hpp"

#include <stdexcept>

namespace prbench {

bool gpu_runtime_available() noexcept { return false; }

std::string gpu_unavailable_reason() {
    return "worker was built without CUDA support";
}

std::string gpu_library_versions_json() { return "{\"cuda_enabled\":false}"; }

std::unique_ptr<IGpuReducer> make_gpu_reducer(const GpuReducerConfig&) {
    throw std::runtime_error(gpu_unavailable_reason());
}

}  // namespace prbench
