#pragma once

#include <cstddef>

#include "prbench/metrics.hpp"
#include "prbench/types.hpp"

namespace prbench {

struct CpuReductionResult {
    Value result;
    double compute_us{0.0};
};

bool cpu_backend_supported(CpuBackendKind backend) noexcept;
CpuReductionResult reduce_cpu(
    const void* data,
    std::size_t count,
    DataType dtype,
    CpuBackendKind backend,
    int threads,
    ReductionOperation operation
);

}  // namespace prbench
