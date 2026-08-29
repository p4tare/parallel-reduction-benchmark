#include "prbench/cpu_backend.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if PRBENCH_HAS_OPENMP
#include <omp.h>
#endif

namespace prbench {
namespace {

template <typename T>
Value to_value(T value) {
    if constexpr (std::is_integral_v<T>) {
        return Value(static_cast<std::int64_t>(value));
    } else {
        return Value(static_cast<double>(value));
    }
}

template <typename T, ReductionOperation Op>
constexpr T identity_value() {
    if constexpr (Op == ReductionOperation::Sum) return T{};
    if constexpr (Op == ReductionOperation::Min) return std::numeric_limits<T>::max();
    return std::numeric_limits<T>::lowest();
}

template <ReductionOperation Op, typename T>
inline void combine_scalar(T& dst, T value) {
    if constexpr (Op == ReductionOperation::Sum) dst += value;
    else if constexpr (Op == ReductionOperation::Min) dst = value < dst ? value : dst;
    else dst = value > dst ? value : dst;
}

template <typename T, ReductionOperation Op>
CpuReductionResult reduce_typed_op(const T* data, std::size_t count, CpuBackendKind backend, int threads) {
    using clock = std::chrono::steady_clock;
    T result = identity_value<T, Op>();
    const auto start = clock::now();

    if (backend == CpuBackendKind::Sequential) {
        for (std::size_t i = 0; i < count; ++i) combine_scalar<Op>(result, data[i]);
    } else if (backend == CpuBackendKind::OpenMP) {
#if PRBENCH_HAS_OPENMP
        omp_set_dynamic(0);
        if constexpr (Op == ReductionOperation::Sum) {
#pragma omp parallel for reduction(+ : result) schedule(static) num_threads(threads)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) result += data[i];
        } else if constexpr (Op == ReductionOperation::Min) {
#pragma omp parallel for reduction(min : result) schedule(static) num_threads(threads)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
                result = data[i] < result ? data[i] : result;
            }
        } else {
#pragma omp parallel for reduction(max : result) schedule(static) num_threads(threads)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
                result = data[i] > result ? data[i] : result;
            }
        }
#else
        throw std::runtime_error("OpenMP backend requested but worker was built without OpenMP");
#endif
    } else if (backend == CpuBackendKind::OpenMPSimd) {
#if PRBENCH_HAS_OPENMP
        omp_set_dynamic(0);
        if constexpr (Op == ReductionOperation::Sum) {
#pragma omp parallel for simd reduction(+ : result) schedule(static) num_threads(threads)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) result += data[i];
        } else if constexpr (Op == ReductionOperation::Min) {
#pragma omp parallel for simd reduction(min : result) schedule(static) num_threads(threads)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
                result = data[i] < result ? data[i] : result;
            }
        } else {
#pragma omp parallel for simd reduction(max : result) schedule(static) num_threads(threads)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
                result = data[i] > result ? data[i] : result;
            }
        }
#else
        throw std::runtime_error("OpenMP SIMD backend requested but worker was built without OpenMP");
#endif
    } else {
        throw std::invalid_argument("invalid CPU backend for reduction");
    }

    const auto stop = clock::now();
    const double us = std::chrono::duration<double, std::micro>(stop - start).count();
    return CpuReductionResult{to_value(result), us};
}

template <typename T>
CpuReductionResult reduce_typed(
    const T* data,
    std::size_t count,
    CpuBackendKind backend,
    int threads,
    ReductionOperation operation
) {
    switch (operation) {
        case ReductionOperation::Sum:
            return reduce_typed_op<T, ReductionOperation::Sum>(data, count, backend, threads);
        case ReductionOperation::Min:
            return reduce_typed_op<T, ReductionOperation::Min>(data, count, backend, threads);
        case ReductionOperation::Max:
            return reduce_typed_op<T, ReductionOperation::Max>(data, count, backend, threads);
    }
    throw std::logic_error("unreachable reduction operation");
}

}  // namespace

bool cpu_backend_supported(CpuBackendKind backend) noexcept {
    if (backend == CpuBackendKind::Sequential) return true;
#if PRBENCH_HAS_OPENMP
    return backend == CpuBackendKind::OpenMP || backend == CpuBackendKind::OpenMPSimd;
#else
    return false;
#endif
}

CpuReductionResult reduce_cpu(
    const void* data,
    std::size_t count,
    DataType dtype,
    CpuBackendKind backend,
    int threads,
    ReductionOperation operation
) {
    switch (dtype) {
        case DataType::Int32:
            return reduce_typed(static_cast<const std::int32_t*>(data), count, backend, threads, operation);
        case DataType::Int64:
            return reduce_typed(static_cast<const std::int64_t*>(data), count, backend, threads, operation);
        case DataType::Float32:
            return reduce_typed(static_cast<const float*>(data), count, backend, threads, operation);
        case DataType::Float64:
            return reduce_typed(static_cast<const double*>(data), count, backend, threads, operation);
    }
    throw std::logic_error("unreachable");
}

}  // namespace prbench
