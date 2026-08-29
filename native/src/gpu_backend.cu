#include "prbench/gpu_backend.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace prbench {
namespace {

// Variadic on purpose: CUDA calls wrapped here may contain template argument lists
// such as cub_reduce<T, Op>(...), whose comma is visible to the C preprocessor.
// A one-argument macro would incorrectly parse that as two macro arguments.
#define CUDA_CHECK(...)                                                                             \
    do {                                                                                            \
        cudaError_t _err = (__VA_ARGS__);                                                           \
        if (_err != cudaSuccess) {                                                                  \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) +     \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                                           \
    } while (0)

using host_clock = std::chrono::steady_clock;

template <typename T>
__host__ __device__ constexpr T reduction_max_finite() {
    // Keep the identity implementation fully device-safe on CUDA 12.x.
    // std::numeric_limits<T>::max()/lowest() are constexpr host functions in
    // some libstdc++/NVCC combinations and trigger NVCC diagnostic #20013-D
    // when called from a __host__ __device__ function.  These exact constants
    // avoid relaxed-constexpr compiler flags and keep host/device semantics equal.
    if constexpr (std::is_same_v<T, std::int32_t>) {
        return static_cast<T>(2147483647);
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return static_cast<T>(9223372036854775807LL);
    } else if constexpr (std::is_same_v<T, float>) {
        return 0x1.fffffep+127f;
    } else if constexpr (std::is_same_v<T, double>) {
        return 0x1.fffffffffffffp+1023;
    } else {
        static_assert(!sizeof(T), "unsupported CUDA reduction type");
    }
}

template <typename T, ReductionOperation Op>
__host__ __device__ constexpr T reduction_identity() {
    if constexpr (Op == ReductionOperation::Sum) {
        return T{};
    } else if constexpr (Op == ReductionOperation::Min) {
        return reduction_max_finite<T>();
    } else {
        return -reduction_max_finite<T>();
    }
}

template <typename T, ReductionOperation Op>
__host__ __device__ inline T reduction_combine(T a, T b) {
    if constexpr (Op == ReductionOperation::Sum) return a + b;
    if constexpr (Op == ReductionOperation::Min) return b < a ? b : a;
    return b > a ? b : a;
}

__device__ inline void device_atomic_add(std::int32_t* address, std::int32_t value) {
    atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(value));
}

__device__ inline void device_atomic_add(std::int64_t* address, std::int64_t value) {
    atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
}

__device__ inline void device_atomic_add(float* address, float value) {
    atomicAdd(address, value);
}

__device__ inline void device_atomic_add(double* address, double value) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    auto* ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *ull;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(ull, assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

template <ReductionOperation Op>
__device__ inline void device_atomic_ordered(std::int32_t* address, std::int32_t value) {
    auto* raw = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *raw;
    while (true) {
        const unsigned int assumed = old;
        const auto current = static_cast<std::int32_t>(assumed);
        const auto desired = reduction_combine<std::int32_t, Op>(current, value);
        if (desired == current) return;
        old = atomicCAS(raw, assumed, static_cast<unsigned int>(desired));
        if (old == assumed) return;
    }
}

template <ReductionOperation Op>
__device__ inline void device_atomic_ordered(std::int64_t* address, std::int64_t value) {
    auto* raw = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *raw;
    while (true) {
        const unsigned long long assumed = old;
        const auto current = static_cast<std::int64_t>(assumed);
        const auto desired = reduction_combine<std::int64_t, Op>(current, value);
        if (desired == current) return;
        old = atomicCAS(raw, assumed, static_cast<unsigned long long>(desired));
        if (old == assumed) return;
    }
}

template <ReductionOperation Op>
__device__ inline void device_atomic_ordered(float* address, float value) {
    auto* raw = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *raw;
    while (true) {
        const unsigned int assumed = old;
        const float current = __int_as_float(static_cast<int>(assumed));
        const float desired = reduction_combine<float, Op>(current, value);
        if (desired == current) return;
        old = atomicCAS(raw, assumed, static_cast<unsigned int>(__float_as_int(desired)));
        if (old == assumed) return;
    }
}

template <ReductionOperation Op>
__device__ inline void device_atomic_ordered(double* address, double value) {
    auto* raw = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *raw;
    while (true) {
        const unsigned long long assumed = old;
        const double current = __longlong_as_double(static_cast<long long>(assumed));
        const double desired = reduction_combine<double, Op>(current, value);
        if (desired == current) return;
        old = atomicCAS(raw, assumed, static_cast<unsigned long long>(__double_as_longlong(desired)));
        if (old == assumed) return;
    }
}

template <typename T, ReductionOperation Op>
__device__ inline void device_atomic_combine(T* address, T value) {
    if constexpr (Op == ReductionOperation::Sum) device_atomic_add(address, value);
    else device_atomic_ordered<Op>(address, value);
}

template <typename T, ReductionOperation Op>
__device__ inline T warp_reduce(T value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = reduction_combine<T, Op>(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename T, ReductionOperation Op>
__global__ void global_atomic_kernel(const T* input, T* output, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride) {
        device_atomic_combine<T, Op>(output, input[i]);
    }
}

template <typename T, ReductionOperation Op>
__global__ void shared_naive_kernel(const T* input, T* output, std::size_t n) {
    extern __shared__ unsigned char raw[];
    T* shared = reinterpret_cast<T*>(raw);
    const unsigned tid = threadIdx.x;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    T local = reduction_identity<T, Op>();
    for (std::size_t i = idx; i < n; i += stride) local = reduction_combine<T, Op>(local, input[i]);
    shared[tid] = local;
    __syncthreads();

    // Intentionally simple interleaved tree. This is a didactic baseline, not the optimized kernel.
    for (unsigned s = 1; s < blockDim.x; s <<= 1) {
        if ((tid % (2 * s)) == 0) shared[tid] = reduction_combine<T, Op>(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    if (tid == 0) device_atomic_combine<T, Op>(output, shared[0]);
}

template <typename T, ReductionOperation Op>
__global__ void warp_atomic_kernel(const T* input, T* output, std::size_t n) {
    __shared__ T warp_values[32];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    T local = reduction_identity<T, Op>();
    for (std::size_t i = idx; i < n; i += stride) local = reduction_combine<T, Op>(local, input[i]);
    local = warp_reduce<T, Op>(local);
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();

    if (warp == 0) {
        const unsigned warp_count = (blockDim.x + 31u) / 32u;
        T block_value = lane < warp_count ? warp_values[lane] : reduction_identity<T, Op>();
        block_value = warp_reduce<T, Op>(block_value);
        if (lane == 0) device_atomic_combine<T, Op>(output, block_value);
    }
}

template <typename T, ReductionOperation Op>
__global__ void block_reduce_kernel(const T* input, T* block_values, std::size_t n) {
    __shared__ T warp_values[32];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    T local = reduction_identity<T, Op>();
    for (std::size_t i = idx; i < n; i += stride) local = reduction_combine<T, Op>(local, input[i]);
    local = warp_reduce<T, Op>(local);
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();
    if (warp == 0) {
        const unsigned warp_count = (blockDim.x + 31u) / 32u;
        T block_value = lane < warp_count ? warp_values[lane] : reduction_identity<T, Op>();
        block_value = warp_reduce<T, Op>(block_value);
        if (lane == 0) block_values[blockIdx.x] = block_value;
    }
}

template <typename T, ReductionOperation Op>
cudaError_t cub_reduce(
    void* temp,
    std::size_t& temp_bytes,
    const T* input,
    T* output,
    std::size_t count,
    cudaStream_t stream
) {
    if constexpr (Op == ReductionOperation::Sum) {
        return cub::DeviceReduce::Sum(temp, temp_bytes, input, output, count, stream);
    } else if constexpr (Op == ReductionOperation::Min) {
        return cub::DeviceReduce::Min(temp, temp_bytes, input, output, count, stream);
    } else {
        return cub::DeviceReduce::Max(temp, temp_bytes, input, output, count, stream);
    }
}

Value make_value(std::int32_t value) { return Value(static_cast<std::int64_t>(value)); }
Value make_value(std::int64_t value) { return Value(value); }
Value make_value(float value) { return Value(static_cast<double>(value)); }
Value make_value(double value) { return Value(value); }

struct EventPair {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~EventPair() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    void record_start(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start, stream)); }
    void record_stop(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(stop, stream)); }
    double elapsed_us() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return static_cast<double>(ms) * 1000.0;
    }
};

class CudaGpuReducer final : public IGpuReducer {
public:
    explicit CudaGpuReducer(const GpuReducerConfig& cfg) : cfg_(cfg) {
        CUDA_CHECK(cudaSetDevice(cfg_.device_id));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg_.device_id));
        sm_count_ = std::max(1, prop.multiProcessorCount);
        element_size_ = data_type_size(cfg_.dtype);

        if (cfg_.transfer_policy == TransferPolicy::AsyncPipeline && cfg_.backend != GpuBackendKind::Cub) {
            throw std::invalid_argument("async_pipeline is currently supported only with the CUB GPU backend");
        }
        if (cfg_.transfer_policy == TransferPolicy::Sync) setup_sync();
        else setup_pipeline();
    }

    ~CudaGpuReducer() override {
        try {
            CUDA_CHECK(cudaSetDevice(cfg_.device_id));
            if (h_output_pinned_) cudaFreeHost(h_output_pinned_);
            for (auto* ptr : h_pipeline_inputs_) if (ptr) cudaFreeHost(ptr);
            for (auto* ptr : d_inputs_) if (ptr) cudaFree(ptr);
            for (auto* ptr : d_outputs_) if (ptr) cudaFree(ptr);
            for (auto* ptr : d_temp_) if (ptr) cudaFree(ptr);
            for (auto* ptr : d_scratch_a_) if (ptr) cudaFree(ptr);
            for (auto* ptr : d_scratch_b_) if (ptr) cudaFree(ptr);
            for (auto stream : streams_) if (stream) cudaStreamDestroy(stream);
            if (h_pipeline_results_) cudaFreeHost(h_pipeline_results_);
        } catch (...) {
            // Destructors must not throw.
        }
    }

    PartialResult reduce(const void* host_data, std::size_t count) override {
        if (count > cfg_.max_elements) {
            throw std::invalid_argument("GPU reducer received more elements than configured max_elements");
        }
        if (count == 0) {
            PartialResult empty;
            empty.result = Value::identity(cfg_.dtype, cfg_.operation);
            empty.device.device_id = cfg_.device_id;
            return empty;
        }
        switch (cfg_.dtype) {
            case DataType::Int32: return reduce_dispatch(static_cast<const std::int32_t*>(host_data), count);
            case DataType::Int64: return reduce_dispatch(static_cast<const std::int64_t*>(host_data), count);
            case DataType::Float32: return reduce_dispatch(static_cast<const float*>(host_data), count);
            case DataType::Float64: return reduce_dispatch(static_cast<const double*>(host_data), count);
        }
        throw std::logic_error("unreachable");
    }

private:
    void setup_sync() {
        streams_.resize(1);
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[0], cudaStreamNonBlocking));
        d_inputs_.resize(1, nullptr);
        d_outputs_.resize(1, nullptr);
        d_temp_.resize(1, nullptr);
        d_temp_bytes_.resize(1, 0);
        d_scratch_a_.resize(1, nullptr);
        d_scratch_b_.resize(1, nullptr);

        CUDA_CHECK(cudaMalloc(&d_inputs_[0], std::max<std::size_t>(1, cfg_.max_elements) * element_size_));
        CUDA_CHECK(cudaMalloc(&d_outputs_[0], element_size_));
        CUDA_CHECK(cudaHostAlloc(&h_output_pinned_, element_size_, cudaHostAllocPortable));
        setup_backend_storage(0, cfg_.max_elements);
    }

    void setup_pipeline() {
        pipeline_chunk_capacity_ = cfg_.pipeline_chunk_elements > 0
            ? std::max<std::size_t>(1, std::min(cfg_.max_elements, cfg_.pipeline_chunk_elements))
            : std::max<std::size_t>(1, (cfg_.max_elements + cfg_.pipeline_chunks - 1) / cfg_.pipeline_chunks);
        pipeline_max_chunks_ =
            std::max<std::size_t>(1, (cfg_.max_elements + pipeline_chunk_capacity_ - 1) / pipeline_chunk_capacity_);
        streams_.resize(cfg_.pipeline_streams);
        d_inputs_.resize(cfg_.pipeline_streams, nullptr);
        d_outputs_.resize(cfg_.pipeline_streams, nullptr);
        d_temp_.resize(cfg_.pipeline_streams, nullptr);
        d_temp_bytes_.resize(cfg_.pipeline_streams, 0);
        d_scratch_a_.resize(cfg_.pipeline_streams, nullptr);
        d_scratch_b_.resize(cfg_.pipeline_streams, nullptr);
        h_pipeline_inputs_.resize(cfg_.pipeline_streams, nullptr);
        for (int s = 0; s < cfg_.pipeline_streams; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&d_inputs_[s], pipeline_chunk_capacity_ * element_size_));
            CUDA_CHECK(cudaMalloc(&d_outputs_[s], element_size_));
            CUDA_CHECK(cudaHostAlloc(&h_pipeline_inputs_[s], pipeline_chunk_capacity_ * element_size_, cudaHostAllocPortable));
            setup_backend_storage(s, pipeline_chunk_capacity_);
        }
        CUDA_CHECK(cudaHostAlloc(
            &h_pipeline_results_, pipeline_max_chunks_ * element_size_,
            cudaHostAllocPortable
        ));
        pipeline_events_.reserve(pipeline_max_chunks_ * 3);
        for (std::size_t i = 0; i < pipeline_max_chunks_ * 3; ++i) {
            pipeline_events_.push_back(std::make_unique<EventPair>());
        }
    }

    void setup_backend_storage(int slot, std::size_t capacity) {
        if (cfg_.backend == GpuBackendKind::Cub) {
            switch (cfg_.dtype) {
                case DataType::Int32: query_cub<std::int32_t>(slot, capacity); break;
                case DataType::Int64: query_cub<std::int64_t>(slot, capacity); break;
                case DataType::Float32: query_cub<float>(slot, capacity); break;
                case DataType::Float64: query_cub<double>(slot, capacity); break;
            }
        } else if (cfg_.backend == GpuBackendKind::TwoPass) {
            const std::size_t blocks = max_grid(capacity);
            CUDA_CHECK(cudaMalloc(&d_scratch_a_[slot], std::max<std::size_t>(1, blocks) * element_size_));
            CUDA_CHECK(cudaMalloc(&d_scratch_b_[slot], std::max<std::size_t>(1, blocks) * element_size_));
        }
    }

    template <typename T, ReductionOperation Op>
    void query_cub_op(int slot, std::size_t capacity) {
        std::size_t bytes = 0;
        CUDA_CHECK(cub_reduce<T, Op>(
            nullptr, bytes,
            static_cast<const T*>(d_inputs_[slot]), static_cast<T*>(d_outputs_[slot]),
            capacity, streams_[slot]
        ));
        d_temp_bytes_[slot] = bytes;
        if (bytes > 0) CUDA_CHECK(cudaMalloc(&d_temp_[slot], bytes));
    }

    template <typename T>
    void query_cub(int slot, std::size_t capacity) {
        switch (cfg_.operation) {
            case ReductionOperation::Sum: query_cub_op<T, ReductionOperation::Sum>(slot, capacity); break;
            case ReductionOperation::Min: query_cub_op<T, ReductionOperation::Min>(slot, capacity); break;
            case ReductionOperation::Max: query_cub_op<T, ReductionOperation::Max>(slot, capacity); break;
        }
    }

    std::size_t max_grid(std::size_t n) const {
        const std::size_t natural = (n + static_cast<std::size_t>(cfg_.block_size) - 1) /
                                    static_cast<std::size_t>(cfg_.block_size);
        return std::max<std::size_t>(1, std::min<std::size_t>(natural, static_cast<std::size_t>(sm_count_) * 32));
    }

    template <typename T>
    PartialResult reduce_dispatch(const T* host_data, std::size_t count) {
        switch (cfg_.operation) {
            case ReductionOperation::Sum: return reduce_dispatch_op<T, ReductionOperation::Sum>(host_data, count);
            case ReductionOperation::Min: return reduce_dispatch_op<T, ReductionOperation::Min>(host_data, count);
            case ReductionOperation::Max: return reduce_dispatch_op<T, ReductionOperation::Max>(host_data, count);
        }
        throw std::logic_error("unreachable operation");
    }

    template <typename T, ReductionOperation Op>
    PartialResult reduce_dispatch_op(const T* host_data, std::size_t count) {
        if (cfg_.transfer_policy == TransferPolicy::AsyncPipeline) {
            return reduce_pipeline<T, Op>(host_data, count);
        }
        if (cfg_.transfer_policy == TransferPolicy::DeviceResident) {
            return reduce_device_resident<T, Op>(host_data, count);
        }
        return reduce_sync<T, Op>(host_data, count);
    }

    template <typename T, ReductionOperation Op>
    PartialResult reduce_sync(const T* host_data, std::size_t count) {
        CUDA_CHECK(cudaSetDevice(cfg_.device_id));
        auto* d_input = static_cast<T*>(d_inputs_[0]);
        auto* d_output = static_cast<T*>(d_outputs_[0]);
        auto* h_output = static_cast<T*>(h_output_pinned_);
        const auto stream = streams_[0];
        DeviceMetrics metrics;
        metrics.device_id = cfg_.device_id;
        metrics.chunks = 1;
        metrics.elements = count;

        EventPair overhead;
        EventPair h2d;
        EventPair kernel;
        EventPair d2h;
        const auto host_start = host_clock::now();

        if (cfg_.backend == GpuBackendKind::GlobalAtomic ||
            cfg_.backend == GpuBackendKind::SharedNaive ||
            cfg_.backend == GpuBackendKind::WarpAtomic) {
            overhead.record_start(stream);
            if constexpr (Op == ReductionOperation::Sum) {
                CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(T), stream));
            } else {
                *h_output = reduction_identity<T, Op>();
                CUDA_CHECK(cudaMemcpyAsync(d_output, h_output, sizeof(T), cudaMemcpyHostToDevice, stream));
            }
            overhead.record_stop(stream);
        }

        h2d.record_start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_input, host_data, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        h2d.record_stop(stream);

        kernel.record_start(stream);
        const std::size_t blocks = max_grid(count);
        if (cfg_.backend == GpuBackendKind::GlobalAtomic) {
            global_atomic_kernel<T, Op><<<static_cast<unsigned>(blocks), cfg_.block_size, 0, stream>>>(d_input, d_output, count);
        } else if (cfg_.backend == GpuBackendKind::SharedNaive) {
            shared_naive_kernel<T, Op><<<static_cast<unsigned>(blocks), cfg_.block_size, cfg_.block_size * sizeof(T), stream>>>(
                d_input, d_output, count
            );
        } else if (cfg_.backend == GpuBackendKind::WarpAtomic) {
            warp_atomic_kernel<T, Op><<<static_cast<unsigned>(blocks), cfg_.block_size, 0, stream>>>(d_input, d_output, count);
        } else if (cfg_.backend == GpuBackendKind::TwoPass) {
            d_output = two_pass<T, Op>(d_input, count, stream, 0);
        } else if (cfg_.backend == GpuBackendKind::Cub) {
            CUDA_CHECK(cub_reduce<T, Op>(d_temp_[0], d_temp_bytes_[0], d_input, d_output, count, stream));
        } else {
            throw std::invalid_argument("invalid GPU backend");
        }
        CUDA_CHECK(cudaGetLastError());
        kernel.record_stop(stream);

        d2h.record_start(stream);
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
        d2h.record_stop(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto host_end = host_clock::now();

        metrics.h2d_us = h2d.elapsed_us();
        metrics.kernel_us = kernel.elapsed_us();
        metrics.d2h_us = d2h.elapsed_us();
        metrics.device_overhead_us =
            (cfg_.backend == GpuBackendKind::GlobalAtomic || cfg_.backend == GpuBackendKind::SharedNaive ||
             cfg_.backend == GpuBackendKind::WarpAtomic)
                ? overhead.elapsed_us()
                : 0.0;
        metrics.total_us = std::chrono::duration<double, std::micro>(host_end - host_start).count();
        return PartialResult{make_value(*h_output), metrics};
    }

    template <typename T, ReductionOperation Op>
    T* two_pass(T* d_input, std::size_t count, cudaStream_t stream, int slot) {
        T* current_input = d_input;
        std::size_t current_count = count;
        T* out_a = static_cast<T*>(d_scratch_a_[slot]);
        T* out_b = static_cast<T*>(d_scratch_b_[slot]);
        bool use_a = true;
        while (current_count > 1) {
            const std::size_t blocks = max_grid(current_count);
            T* current_output = use_a ? out_a : out_b;
            block_reduce_kernel<T, Op><<<static_cast<unsigned>(blocks), cfg_.block_size, 0, stream>>>(
                current_input, current_output, current_count
            );
            CUDA_CHECK(cudaGetLastError());
            current_input = current_output;
            current_count = blocks;
            use_a = !use_a;
        }
        return current_input;
    }

    template <typename T, ReductionOperation Op>
    PartialResult reduce_device_resident(const T* host_data, std::size_t count) {
        CUDA_CHECK(cudaSetDevice(cfg_.device_id));
        auto* d_input = static_cast<T*>(d_inputs_[0]);
        auto* d_output = static_cast<T*>(d_outputs_[0]);

        if (!device_resident_loaded_) {
            CUDA_CHECK(cudaMemcpy(
                d_input,
                host_data,
                count * sizeof(T),
                cudaMemcpyHostToDevice
            ));
            device_resident_loaded_ = true;
        }

        DeviceMetrics metrics;
        metrics.device_id = cfg_.device_id;
        metrics.elements = count;
        metrics.chunks = 1;
        auto* h_output = static_cast<T*>(h_output_pinned_);

        // Event construction is setup/instrumentation overhead, not part of the
        // device-resident solution interval (same convention as reduce_sync).
        EventPair kernel;
        EventPair d2h;
        const auto host_start = host_clock::now();
        kernel.record_start(streams_[0]);
        CUDA_CHECK(cub_reduce<T, Op>(
            d_temp_[0],
            d_temp_bytes_[0],
            d_input,
            d_output,
            count,
            streams_[0]
        ));
        kernel.record_stop(streams_[0]);

        d2h.record_start(streams_[0]);
        CUDA_CHECK(cudaMemcpyAsync(
            h_output,
            d_output,
            sizeof(T),
            cudaMemcpyDeviceToHost,
            streams_[0]
        ));
        d2h.record_stop(streams_[0]);
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));

        const auto host_end = host_clock::now();
        metrics.kernel_us = kernel.elapsed_us();
        metrics.d2h_us = d2h.elapsed_us();
        metrics.total_us = std::chrono::duration<double, std::micro>(host_end - host_start).count();
        return PartialResult{make_value(*h_output), metrics};
    }

    template <typename T, ReductionOperation Op>
    PartialResult reduce_pipeline(const T* host_data, std::size_t count) {
        CUDA_CHECK(cudaSetDevice(cfg_.device_id));
        const std::size_t chunk_size = pipeline_chunk_capacity_;
        const std::size_t chunk_count = (count + chunk_size - 1) / chunk_size;
        if (chunk_count > pipeline_max_chunks_) {
            throw std::logic_error("async pipeline chunk count exceeds prepared capacity");
        }
        auto* host_results = static_cast<T*>(h_pipeline_results_);
        DeviceMetrics metrics;
        metrics.device_id = cfg_.device_id;
        const auto host_start = host_clock::now();
        std::size_t used_chunks = 0;

        for (std::size_t c = 0; c < chunk_count; ++c) {
            const std::size_t offset = c * chunk_size;
            if (offset >= count) break;
            const std::size_t n = std::min(chunk_size, count - offset);
            const int slot = static_cast<int>(c % static_cast<std::size_t>(cfg_.pipeline_streams));
            auto stream = streams_[slot];
            auto* d_input = static_cast<T*>(d_inputs_[slot]);
            auto* d_output = static_cast<T*>(d_outputs_[slot]);
            auto* h_staging = static_cast<T*>(h_pipeline_inputs_[slot]);
            EventPair& h2d = *pipeline_events_[c * 3];
            EventPair& kernel = *pipeline_events_[c * 3 + 1];
            EventPair& d2h = *pipeline_events_[c * 3 + 2];

            if (c >= static_cast<std::size_t>(cfg_.pipeline_streams)) CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto staging_start = host_clock::now();
            std::memcpy(h_staging, host_data + offset, n * sizeof(T));
            const auto staging_end = host_clock::now();
            metrics.device_overhead_us +=
                std::chrono::duration<double, std::micro>(staging_end - staging_start).count();

            h2d.record_start(stream);
            CUDA_CHECK(cudaMemcpyAsync(d_input, h_staging, n * sizeof(T), cudaMemcpyHostToDevice, stream));
            h2d.record_stop(stream);

            kernel.record_start(stream);
            CUDA_CHECK(cub_reduce<T, Op>(d_temp_[slot], d_temp_bytes_[slot], d_input, d_output, n, stream));
            kernel.record_stop(stream);

            d2h.record_start(stream);
            CUDA_CHECK(cudaMemcpyAsync(&host_results[c], d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
            d2h.record_stop(stream);
            ++used_chunks;
        }
        for (auto stream : streams_) CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto host_end = host_clock::now();

        T result = reduction_identity<T, Op>();
        for (std::size_t c = 0; c < used_chunks; ++c) {
            result = reduction_combine<T, Op>(result, host_results[c]);
            metrics.h2d_us += pipeline_events_[c * 3]->elapsed_us();
            metrics.kernel_us += pipeline_events_[c * 3 + 1]->elapsed_us();
            metrics.d2h_us += pipeline_events_[c * 3 + 2]->elapsed_us();
        }
        metrics.chunks = static_cast<std::size_t>(used_chunks);
        metrics.elements = count;
        metrics.total_us = std::chrono::duration<double, std::micro>(host_end - host_start).count();
        return PartialResult{make_value(result), metrics};
    }

    GpuReducerConfig cfg_;
    int sm_count_{1};
    std::size_t element_size_{0};
    std::vector<cudaStream_t> streams_;
    std::vector<void*> d_inputs_;
    std::vector<void*> d_outputs_;
    std::vector<void*> d_temp_;
    std::vector<std::size_t> d_temp_bytes_;
    std::vector<void*> d_scratch_a_;
    std::vector<void*> d_scratch_b_;
    void* h_output_pinned_{nullptr};
    std::vector<void*> h_pipeline_inputs_;
    void* h_pipeline_results_{nullptr};
    std::size_t pipeline_chunk_capacity_{1};
    std::size_t pipeline_max_chunks_{1};
    std::vector<std::unique_ptr<EventPair>> pipeline_events_;
    bool device_resident_loaded_{false};
};

}  // namespace

bool gpu_runtime_available() noexcept {
    int count = 0;
    const auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::string gpu_unavailable_reason() {
    int count = 0;
    const auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        const std::string msg = cudaGetErrorString(err);
        cudaGetLastError();
        return "CUDA runtime unavailable: " + msg;
    }
    if (count == 0) return "CUDA runtime is present but no CUDA devices are visible";
    return "unknown CUDA availability error";
}

std::unique_ptr<IGpuReducer> make_gpu_reducer(const GpuReducerConfig& config) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (config.device_id < 0 || config.device_id >= count) {
        throw std::invalid_argument("invalid CUDA device id: " + std::to_string(config.device_id));
    }
    return std::make_unique<CudaGpuReducer>(config);
}


std::string gpu_library_versions_json() {
    std::string out = "{\"cuda_enabled\":true";
#ifdef CUB_VERSION
    out += ",\"cub_version_integer\":" + std::to_string(CUB_VERSION);
#endif
#ifdef CCCL_VERSION
    out += ",\"cccl_version_integer\":" + std::to_string(CCCL_VERSION);
#endif
#ifdef CUDART_VERSION
    out += ",\"cudart_version_integer\":" + std::to_string(CUDART_VERSION);
#endif
    out += "}";
    return out;
}

}  // namespace prbench
