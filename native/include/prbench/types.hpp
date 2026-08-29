#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

namespace prbench {

enum class DataType { Int32, Int64, Float32, Float64 };
enum class ReductionOperation { Sum, Min, Max };
enum class CpuBackendKind { None, Sequential, OpenMP, OpenMPSimd };
enum class GpuBackendKind { None, GlobalAtomic, SharedNaive, WarpAtomic, TwoPass, Cub };
enum class SchedulerKind {
    CpuOnly,
    GpuOnly,
    GpuStaticEqual,
    GpuStaticProfiled,
    StaticEqual,
    StaticProfiled,
    DynamicFixed,
    DynamicGuided,
    DynamicAdaptive
};
enum class TransferPolicy { Sync, AsyncPipeline };

inline DataType parse_data_type(const std::string& s) {
    if (s == "int32") return DataType::Int32;
    if (s == "int64") return DataType::Int64;
    if (s == "float32") return DataType::Float32;
    if (s == "float64" || s == "double") return DataType::Float64;
    throw std::invalid_argument("unsupported data type: " + s);
}


inline ReductionOperation parse_reduction_operation(const std::string& s) {
    if (s == "sum") return ReductionOperation::Sum;
    if (s == "min") return ReductionOperation::Min;
    if (s == "max") return ReductionOperation::Max;
    throw std::invalid_argument("unsupported reduction operation: " + s);
}

inline CpuBackendKind parse_cpu_backend(const std::string& s) {
    if (s == "none") return CpuBackendKind::None;
    if (s == "sequential") return CpuBackendKind::Sequential;
    if (s == "openmp") return CpuBackendKind::OpenMP;
    if (s == "openmp_simd") return CpuBackendKind::OpenMPSimd;
    throw std::invalid_argument("unsupported CPU backend: " + s);
}

inline GpuBackendKind parse_gpu_backend(const std::string& s) {
    if (s == "none") return GpuBackendKind::None;
    if (s == "global_atomic") return GpuBackendKind::GlobalAtomic;
    if (s == "shared_naive") return GpuBackendKind::SharedNaive;
    if (s == "warp_atomic") return GpuBackendKind::WarpAtomic;
    if (s == "two_pass") return GpuBackendKind::TwoPass;
    if (s == "cub") return GpuBackendKind::Cub;
    throw std::invalid_argument("unsupported GPU backend: " + s);
}

inline SchedulerKind parse_scheduler(const std::string& s) {
    if (s == "cpu_only") return SchedulerKind::CpuOnly;
    if (s == "gpu_only") return SchedulerKind::GpuOnly;
    if (s == "gpu_static_equal") return SchedulerKind::GpuStaticEqual;
    if (s == "gpu_static_profiled") return SchedulerKind::GpuStaticProfiled;
    if (s == "static_equal") return SchedulerKind::StaticEqual;
    if (s == "static_profiled") return SchedulerKind::StaticProfiled;
    if (s == "dynamic_fixed") return SchedulerKind::DynamicFixed;
    if (s == "dynamic_guided") return SchedulerKind::DynamicGuided;
    if (s == "dynamic_adaptive") return SchedulerKind::DynamicAdaptive;
    throw std::invalid_argument("unsupported scheduler: " + s);
}

inline TransferPolicy parse_transfer_policy(const std::string& s) {
    if (s == "sync") return TransferPolicy::Sync;
    if (s == "async_pipeline") return TransferPolicy::AsyncPipeline;
    throw std::invalid_argument("unsupported transfer policy: " + s);
}

struct Value {
    std::variant<std::int64_t, double> storage{std::int64_t{0}};

    static Value identity(DataType type, ReductionOperation operation) {
        if (type == DataType::Int32) {
            if (operation == ReductionOperation::Min) return Value{static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())};
            if (operation == ReductionOperation::Max) return Value{static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::lowest())};
            return Value{std::int64_t{0}};
        }
        if (type == DataType::Int64) {
            if (operation == ReductionOperation::Min) return Value{std::numeric_limits<std::int64_t>::max()};
            if (operation == ReductionOperation::Max) return Value{std::numeric_limits<std::int64_t>::lowest()};
            return Value{std::int64_t{0}};
        }
        if (type == DataType::Float32) {
            if (operation == ReductionOperation::Min) return Value{static_cast<double>(std::numeric_limits<float>::max())};
            if (operation == ReductionOperation::Max) return Value{static_cast<double>(std::numeric_limits<float>::lowest())};
            return Value{0.0};
        }
        if (operation == ReductionOperation::Min) return Value{std::numeric_limits<double>::max()};
        if (operation == ReductionOperation::Max) return Value{std::numeric_limits<double>::lowest()};
        return Value{0.0};
    }

    static Value zero(DataType type) { return identity(type, ReductionOperation::Sum); }

    Value() = default;
    explicit Value(std::int64_t v) : storage(v) {}
    explicit Value(double v) : storage(v) {}

    void combine(const Value& other, DataType type, ReductionOperation operation) {
        if (type == DataType::Int32 || type == DataType::Int64) {
            const auto a = std::get<std::int64_t>(storage);
            const auto b = std::get<std::int64_t>(other.storage);
            if (operation == ReductionOperation::Sum) storage = a + b;
            else if (operation == ReductionOperation::Min) storage = std::min(a, b);
            else storage = std::max(a, b);
        } else if (type == DataType::Float32) {
            const float a = static_cast<float>(std::get<double>(storage));
            const float b = static_cast<float>(std::get<double>(other.storage));
            if (operation == ReductionOperation::Sum) storage = static_cast<double>(a + b);
            else if (operation == ReductionOperation::Min) storage = static_cast<double>(std::min(a, b));
            else storage = static_cast<double>(std::max(a, b));
        } else {
            const double a = std::get<double>(storage);
            const double b = std::get<double>(other.storage);
            if (operation == ReductionOperation::Sum) storage = a + b;
            else if (operation == ReductionOperation::Min) storage = std::min(a, b);
            else storage = std::max(a, b);
        }
    }

    void add(const Value& other, DataType type) { combine(other, type, ReductionOperation::Sum); }

    std::string json_literal() const {
        if (std::holds_alternative<std::int64_t>(storage)) {
            return std::to_string(std::get<std::int64_t>(storage));
        }
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::max_digits10)
            << std::get<double>(storage);
        return oss.str();
    }
};

inline std::size_t data_type_size(DataType type) {
    switch (type) {
        case DataType::Int32: return sizeof(std::int32_t);
        case DataType::Int64: return sizeof(std::int64_t);
        case DataType::Float32: return sizeof(float);
        case DataType::Float64: return sizeof(double);
    }
    throw std::logic_error("unreachable");
}

}  // namespace prbench
