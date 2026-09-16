#include "prbench/integrated_strategy.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if PRBENCH_HAS_LIBNUMA
#include <numa.h>
#endif

#include "prbench/affinity.hpp"
#include "prbench/cpu_backend.hpp"

namespace prbench {
namespace {

using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(...) do { \
    const cudaError_t _e = (__VA_ARGS__); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
    } \
} while (0)

template <class F>
double timed_us(F&& f) {
    const auto begin = Clock::now();
    f();
    return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

template <typename T>
Value to_value(T value) {
    if constexpr (std::is_integral_v<T>) return Value(static_cast<std::int64_t>(value));
    return Value(static_cast<double>(value));
}

template <typename T>
T identity(ReductionOperation op) {
    if (op == ReductionOperation::Sum) return T{0};
    if (op == ReductionOperation::Min) return std::numeric_limits<T>::max();
    return std::numeric_limits<T>::lowest();
}

template <typename T>
T combine(T a, T b, ReductionOperation op) {
    if (op == ReductionOperation::Sum) return a + b;
    if (op == ReductionOperation::Min) return std::min(a, b);
    return std::max(a, b);
}

template <typename T>
void read_exact(std::ifstream& input, T* dst, std::size_t n) {
    input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n * sizeof(T)));
    if (input.gcount() != static_cast<std::streamsize>(n * sizeof(T))) {
        throw std::runtime_error("short read from dataset");
    }
}

template <typename T>
void read_file(const std::filesystem::path& path, T* dst, std::size_t n) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset: " + path.string());
    read_exact(input, dst, n);
}

template <typename T>
void cub_query(
    const T* input,
    T* output,
    std::size_t n,
    ReductionOperation op,
    cudaStream_t stream,
    void*& temp,
    std::size_t& temp_bytes
) {
    temp_bytes = 0;
    if (op == ReductionOperation::Sum) {
        CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, output, n, stream));
    } else if (op == ReductionOperation::Min) {
        CUDA_CHECK(cub::DeviceReduce::Min(nullptr, temp_bytes, input, output, n, stream));
    } else {
        CUDA_CHECK(cub::DeviceReduce::Max(nullptr, temp_bytes, input, output, n, stream));
    }
    if (temp_bytes) CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
}

template <typename T>
void cub_reduce(
    void* temp,
    std::size_t temp_bytes,
    const T* input,
    T* output,
    std::size_t n,
    ReductionOperation op,
    cudaStream_t stream
) {
    if (op == ReductionOperation::Sum) {
        CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, input, output, n, stream));
    } else if (op == ReductionOperation::Min) {
        CUDA_CHECK(cub::DeviceReduce::Min(temp, temp_bytes, input, output, n, stream));
    } else {
        CUDA_CHECK(cub::DeviceReduce::Max(temp, temp_bytes, input, output, n, stream));
    }
}

class IntegratedBase : public IReductionStrategy {
public:
    double warmup_median_us() const noexcept override { return warmup_median_us_; }
    const PrepareMetrics& prepare_metrics() const noexcept override { return prepare_metrics_; }

protected:
    void run_warmups(int runs) {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(runs));
        for (int i = 0; i < runs; ++i) samples.push_back(run_once().e2e_us);
        warmup_median_us_ = median(std::move(samples));
    }

    PrepareMetrics prepare_metrics_;

private:
    double warmup_median_us_{0.0};
};

struct HostBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool cuda_allocated{false};
    bool numa_allocated{false};
    bool registered{false};

    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&& other) noexcept { *this = std::move(other); }
    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this == &other) return *this;
        release();
        ptr = other.ptr;
        bytes = other.bytes;
        cuda_allocated = other.cuda_allocated;
        numa_allocated = other.numa_allocated;
        registered = other.registered;
        other.ptr = nullptr;
        other.bytes = 0;
        other.cuda_allocated = false;
        other.numa_allocated = false;
        other.registered = false;
        return *this;
    }
    ~HostBuffer() { release(); }

    void release() noexcept {
        if (!ptr) return;
        if (registered) cudaHostUnregister(ptr);
        if (cuda_allocated) {
            cudaFreeHost(ptr);
        } else if (numa_allocated) {
#if PRBENCH_HAS_LIBNUMA
            numa_free(ptr, bytes);
#endif
        } else {
            std::free(ptr);
        }
        ptr = nullptr;
    }
};

HostBuffer pinned_buffer(std::size_t bytes, int numa_node, bool& numa_applied) {
    HostBuffer out;
    out.bytes = bytes;
#if PRBENCH_HAS_LIBNUMA
    if (numa_node >= 0 && numa_available() >= 0) {
        out.ptr = numa_alloc_onnode(bytes, numa_node);
        if (!out.ptr) throw std::runtime_error("numa_alloc_onnode failed");
        out.numa_allocated = true;
        CUDA_CHECK(cudaHostRegister(out.ptr, bytes, cudaHostRegisterPortable));
        out.registered = true;
        numa_applied = true;
        return out;
    }
#else
    (void)numa_node;
#endif
    CUDA_CHECK(cudaHostAlloc(&out.ptr, bytes, cudaHostAllocPortable));
    out.cuda_allocated = true;
    return out;
}

struct Range {
    std::size_t offset{0};
    std::size_t count{0};
};

std::vector<Range> partition(std::size_t offset, std::size_t count, std::size_t parts) {
    std::vector<Range> ranges(parts);
    const std::size_t base = parts ? count / parts : 0;
    const std::size_t rem = parts ? count % parts : 0;
    std::size_t pos = offset;
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t n = base + (i < rem ? 1 : 0);
        ranges[i] = {pos, n};
        pos += n;
    }
    return ranges;
}

struct DeviceFeatureSet {
    int managed{0};
    int concurrent_managed{0};
    int pageable{0};
    int map_host{0};
};

DeviceFeatureSet features(int device) {
    DeviceFeatureSet out;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    out.map_host = prop.canMapHostMemory;
    CUDA_CHECK(cudaDeviceGetAttribute(&out.managed, cudaDevAttrManagedMemory, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&out.concurrent_managed, cudaDevAttrConcurrentManagedAccess, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&out.pageable, cudaDevAttrPageableMemoryAccess, device));
    return out;
}

template <typename T>
class SingleGpuMemoryStrategy final : public IntegratedBase {
public:
    explicit SingleGpuMemoryStrategy(const WorkerConfig& cfg) : cfg_(cfg), path_(cfg.memory_path) {
        if (cfg_.gpu_ids.size() != 1) {
            throw std::invalid_argument("integrated single-GPU memory path requires exactly one GPU");
        }
        device_ = cfg_.gpu_ids.front();
    }

    ~SingleGpuMemoryStrategy() override { release(); }

    void prepare(int warmup_runs) override {
        CUDA_CHECK(cudaSetDevice(device_));
        setup();
        prepare_metrics_.partition = {{"gpu", 0, device_, 0, cfg_.count}};
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        if (path_ == "chunked_sync" || path_ == "pinned_direct") return run_chunked();
        if (path_ == "zero_copy" || path_ == "hmm_system") return run_direct_host();
        if (path_ == "managed_fault" || path_ == "managed_prefetch" || path_ == "managed_advised") {
            return run_managed();
        }
        return run_full_device();
    }

private:
    void setup() {
        const std::size_t bytes = cfg_.count * sizeof(T);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaHostAlloc(&h_output_, sizeof(T), cudaHostAllocPortable));
        CUDA_CHECK(cudaMalloc(&d_output_, sizeof(T)));

        if (path_ == "zero_copy") {
            host_.resize(cfg_.count);
            read_file(cfg_.dataset_path, host_.data(), cfg_.count);
            CUDA_CHECK(cudaHostRegister(
                host_.data(), bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
            host_registered_ = true;
            CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped_), host_.data(), 0));
            cub_query(mapped_, d_output_, cfg_.count, cfg_.operation, stream_, temp_, temp_bytes_);
            capacity_ = cfg_.count;
            return;
        }

        if (path_ == "hmm_system") {
            host_.resize(cfg_.count);
            read_file(cfg_.dataset_path, host_.data(), cfg_.count);
            cub_query(host_.data(), d_output_, cfg_.count, cfg_.operation, stream_, temp_, temp_bytes_);
            capacity_ = cfg_.count;
            return;
        }

        if (path_ == "managed_fault" || path_ == "managed_prefetch" || path_ == "managed_advised") {
            CUDA_CHECK(cudaMallocManaged(&managed_, bytes));
            read_file(cfg_.dataset_path, managed_, cfg_.count);
            if (path_ == "managed_advised") {
                CUDA_CHECK(cudaMemAdvise(managed_, bytes, cudaMemAdviseSetReadMostly, device_));
                CUDA_CHECK(cudaMemAdvise(managed_, bytes, cudaMemAdviseSetAccessedBy, device_));
            }
            cub_query(managed_, d_output_, cfg_.count, cfg_.operation, stream_, temp_, temp_bytes_);
            capacity_ = cfg_.count;
            return;
        }

        host_.resize(cfg_.count);
        read_file(cfg_.dataset_path, host_.data(), cfg_.count);
        if (path_ == "pinned_direct") {
            CUDA_CHECK(cudaHostRegister(host_.data(), bytes, cudaHostRegisterPortable));
            host_registered_ = true;
        }

        if (path_ == "chunked_sync" || path_ == "pinned_direct") {
            capacity_ = std::min(cfg_.count, cfg_.chunk_size);
        } else {
            capacity_ = cfg_.count;
        }
        CUDA_CHECK(cudaMalloc(&d_input_, std::max<std::size_t>(1, capacity_) * sizeof(T)));
        cub_query(d_input_, d_output_, std::max<std::size_t>(1, capacity_), cfg_.operation, stream_, temp_, temp_bytes_);

        if (path_ == "device_resident") {
            const double upload_us = timed_us([&] {
                CUDA_CHECK(cudaMemcpyAsync(d_input_, host_.data(), bytes, cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            });
            resident_loaded_ = true;
            prepare_metrics_.calibration_samples.push_back({
                "gpu_device_resident_upload", 0, device_, cfg_.count, upload_us,
                static_cast<double>(cfg_.count) / std::max(upload_us, 1e-6) * 1e6
            });
        }
        if (cfg_.use_cuda_graphs && (path_ == "explicit_sync" || path_ == "device_resident")) {
            capture_graph();
        }
    }

    void capture_graph() {
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < cfg_.reuse_count; ++k) {
            cub_reduce(temp_, temp_bytes_, d_input_, d_output_, cfg_.count, cfg_.operation, stream_);
        }
        CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, sizeof(T), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    }

    IterationMetrics run_full_device() {
        CUDA_CHECK(cudaSetDevice(device_));
        DeviceMetrics dm;
        dm.device_id = device_;
        dm.chunks = static_cast<std::size_t>(cfg_.reuse_count);
        dm.elements = cfg_.count * static_cast<std::size_t>(cfg_.reuse_count);
        const auto begin = Clock::now();

        if (path_ == "explicit_sync") {
            dm.h2d_us = timed_us([&] {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_input_, host_.data(), cfg_.count * sizeof(T), cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            });
            dm.h2d_bytes = cfg_.count * sizeof(T);
        }

        if (cfg_.use_cuda_graphs) {
            dm.kernel_us = timed_us([&] {
                CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            });
            dm.d2h_bytes = sizeof(T);
        } else {
            dm.kernel_us = timed_us([&] {
                for (int k = 0; k < cfg_.reuse_count; ++k) {
                    cub_reduce(temp_, temp_bytes_, d_input_, d_output_, cfg_.count, cfg_.operation, stream_);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            });
            dm.d2h_us = timed_us([&] {
                CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, sizeof(T), cudaMemcpyDeviceToHost, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            });
            dm.d2h_bytes = sizeof(T);
        }
        dm.total_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        IterationMetrics out;
        out.result = to_value(*h_output_);
        out.gpus.push_back(dm);
        out.e2e_us = dm.total_us;
        return out;
    }

    IterationMetrics run_chunked() {
        CUDA_CHECK(cudaSetDevice(device_));
        DeviceMetrics dm;
        dm.device_id = device_;
        T final = identity<T>(cfg_.operation);
        const auto begin = Clock::now();
        for (int reuse = 0; reuse < cfg_.reuse_count; ++reuse) {
            T merged = identity<T>(cfg_.operation);
            for (std::size_t offset = 0; offset < cfg_.count; offset += capacity_) {
                const std::size_t n = std::min(capacity_, cfg_.count - offset);
                dm.h2d_us += timed_us([&] {
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_input_, host_.data() + offset, n * sizeof(T), cudaMemcpyHostToDevice, stream_));
                    CUDA_CHECK(cudaStreamSynchronize(stream_));
                });
                dm.h2d_bytes += n * sizeof(T);
                dm.kernel_us += timed_us([&] {
                    cub_reduce(temp_, temp_bytes_, d_input_, d_output_, n, cfg_.operation, stream_);
                    CUDA_CHECK(cudaStreamSynchronize(stream_));
                });
                dm.d2h_us += timed_us([&] {
                    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, sizeof(T), cudaMemcpyDeviceToHost, stream_));
                    CUDA_CHECK(cudaStreamSynchronize(stream_));
                });
                dm.d2h_bytes += sizeof(T);
                merged = combine(merged, *h_output_, cfg_.operation);
                ++dm.chunks;
            }
            final = merged;
        }
        dm.elements = cfg_.count * static_cast<std::size_t>(cfg_.reuse_count);
        dm.total_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        IterationMetrics out;
        out.result = to_value(final);
        out.gpus.push_back(dm);
        out.e2e_us = dm.total_us;
        return out;
    }

    IterationMetrics run_direct_host() {
        CUDA_CHECK(cudaSetDevice(device_));
        DeviceMetrics dm;
        dm.device_id = device_;
        const T* input = path_ == "zero_copy" ? mapped_ : host_.data();
        const auto begin = Clock::now();
        dm.kernel_us = timed_us([&] {
            for (int k = 0; k < cfg_.reuse_count; ++k) {
                cub_reduce(temp_, temp_bytes_, input, d_output_, cfg_.count, cfg_.operation, stream_);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        });
        dm.d2h_us = timed_us([&] {
            CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, sizeof(T), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        });
        dm.remote_host_read_bytes = static_cast<std::uint64_t>(cfg_.count * sizeof(T)) * cfg_.reuse_count;
        dm.d2h_bytes = sizeof(T);
        dm.chunks = static_cast<std::size_t>(cfg_.reuse_count);
        dm.elements = cfg_.count * static_cast<std::size_t>(cfg_.reuse_count);
        dm.total_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        IterationMetrics out;
        out.result = to_value(*h_output_);
        out.gpus.push_back(dm);
        out.e2e_us = dm.total_us;
        return out;
    }

    IterationMetrics run_managed() {
        CUDA_CHECK(cudaSetDevice(device_));
        const std::size_t bytes = cfg_.count * sizeof(T);
        // Establish the CPU as the owner immediately before the measured interval.
        CUDA_CHECK(cudaMemPrefetchAsync(managed_, bytes, cudaCpuDeviceId, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        DeviceMetrics dm;
        dm.device_id = device_;
        const auto begin = Clock::now();
        if (path_ == "managed_prefetch") {
            dm.h2d_us = timed_us([&] {
                CUDA_CHECK(cudaMemPrefetchAsync(managed_, bytes, device_, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            });
            dm.h2d_bytes = bytes;
        }
        dm.kernel_us = timed_us([&] {
            for (int k = 0; k < cfg_.reuse_count; ++k) {
                cub_reduce(temp_, temp_bytes_, managed_, d_output_, cfg_.count, cfg_.operation, stream_);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        });
        dm.d2h_us = timed_us([&] {
            CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, sizeof(T), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        });
        dm.d2h_bytes = sizeof(T);
        dm.chunks = static_cast<std::size_t>(cfg_.reuse_count);
        dm.elements = cfg_.count * static_cast<std::size_t>(cfg_.reuse_count);
        dm.total_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        IterationMetrics out;
        out.result = to_value(*h_output_);
        out.gpus.push_back(dm);
        out.e2e_us = dm.total_us;
        return out;
    }

    void release() noexcept {
        if (device_ >= 0) cudaSetDevice(device_);
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (temp_) cudaFree(temp_);
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        if (h_output_) cudaFreeHost(h_output_);
        if (managed_) cudaFree(managed_);
        if (host_registered_ && !host_.empty()) cudaHostUnregister(host_.data());
        if (stream_) cudaStreamDestroy(stream_);
        graph_exec_ = nullptr;
        graph_ = nullptr;
        temp_ = nullptr;
        d_input_ = nullptr;
        d_output_ = nullptr;
        h_output_ = nullptr;
        managed_ = nullptr;
        stream_ = nullptr;
        host_registered_ = false;
    }

    WorkerConfig cfg_;
    std::string path_;
    int device_{-1};
    std::size_t capacity_{0};
    std::vector<T> host_;
    T* mapped_{nullptr};
    T* managed_{nullptr};
    T* d_input_{nullptr};
    T* d_output_{nullptr};
    T* h_output_{nullptr};
    void* temp_{nullptr};
    std::size_t temp_bytes_{0};
    cudaStream_t stream_{};
    cudaGraph_t graph_{};
    cudaGraphExec_t graph_exec_{};
    bool host_registered_{false};
    bool resident_loaded_{false};
};

template <typename T>
struct GpuFilePipeline {
    struct Slot {
        cudaStream_t stream{};
        HostBuffer staging;
        T* d_input{nullptr};
        T* d_output{nullptr};
        T* h_output{nullptr};
        void* temp{nullptr};
        std::size_t temp_bytes{0};
        bool pending{false};
    };

    int device{-1};
    int numa_node{-1};
    std::size_t capacity{0};
    ReductionOperation op{ReductionOperation::Sum};
    std::vector<Slot> slots;
    bool numa_applied{false};

    GpuFilePipeline(int dev, int node, std::size_t cap, int stream_count, ReductionOperation operation)
        : device(dev), numa_node(node), capacity(cap), op(operation), slots(static_cast<std::size_t>(stream_count)) {
        CUDA_CHECK(cudaSetDevice(device));
        for (auto& slot : slots) {
            slot.staging = pinned_buffer(capacity * sizeof(T), numa_node, numa_applied);
            CUDA_CHECK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&slot.d_input, capacity * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&slot.d_output, sizeof(T)));
            CUDA_CHECK(cudaHostAlloc(&slot.h_output, sizeof(T), cudaHostAllocPortable));
            cub_query(slot.d_input, slot.d_output, capacity, op, slot.stream, slot.temp, slot.temp_bytes);
        }
    }

    ~GpuFilePipeline() {
        cudaSetDevice(device);
        for (auto& slot : slots) {
            if (slot.temp) cudaFree(slot.temp);
            if (slot.d_input) cudaFree(slot.d_input);
            if (slot.d_output) cudaFree(slot.d_output);
            if (slot.h_output) cudaFreeHost(slot.h_output);
            if (slot.stream) cudaStreamDestroy(slot.stream);
        }
    }
};

template <typename T>
T reduce_gpu_file_range(
    const WorkerConfig& cfg,
    Range range,
    GpuFilePipeline<T>& pipe,
    DeviceMetrics& dm
) {
    CUDA_CHECK(cudaSetDevice(pipe.device));
    dm.device_id = pipe.device;
    dm.numa_requested = pipe.numa_node >= 0;
    dm.numa_applied = pipe.numa_applied;
#if PRBENCH_HAS_LIBNUMA
    if (pipe.numa_node >= 0 && numa_available() >= 0) (void)numa_run_on_node(pipe.numa_node);
#endif

    T final = identity<T>(cfg.operation);
    for (int reuse = 0; reuse < cfg.reuse_count; ++reuse) {
        std::ifstream input(cfg.dataset_path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open dataset for GPU file stream");
        input.seekg(static_cast<std::streamoff>(range.offset * sizeof(T)), std::ios::beg);
        if (!input) throw std::runtime_error("cannot seek GPU file stream");
        T merged = identity<T>(cfg.operation);
        std::size_t chunk_index = 0;
        for (std::size_t done = 0; done < range.count; done += pipe.capacity, ++chunk_index) {
            auto& slot = pipe.slots[chunk_index % pipe.slots.size()];
            if (slot.pending) {
                CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                merged = combine(merged, *slot.h_output, cfg.operation);
                slot.pending = false;
            }
            const std::size_t n = std::min(pipe.capacity, range.count - done);
            dm.storage_read_us += timed_us([&] { read_exact(input, static_cast<T*>(slot.staging.ptr), n); });
            dm.storage_read_bytes += n * sizeof(T);

            if (pipe.slots.size() == 1) {
                dm.h2d_us += timed_us([&] {
                    CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.staging.ptr, n * sizeof(T), cudaMemcpyHostToDevice, slot.stream));
                    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                });
                dm.kernel_us += timed_us([&] {
                    cub_reduce(slot.temp, slot.temp_bytes, slot.d_input, slot.d_output, n, cfg.operation, slot.stream);
                    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                });
                dm.d2h_us += timed_us([&] {
                    CUDA_CHECK(cudaMemcpyAsync(slot.h_output, slot.d_output, sizeof(T), cudaMemcpyDeviceToHost, slot.stream));
                    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                });
                merged = combine(merged, *slot.h_output, cfg.operation);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.staging.ptr, n * sizeof(T), cudaMemcpyHostToDevice, slot.stream));
                cub_reduce(slot.temp, slot.temp_bytes, slot.d_input, slot.d_output, n, cfg.operation, slot.stream);
                CUDA_CHECK(cudaMemcpyAsync(slot.h_output, slot.d_output, sizeof(T), cudaMemcpyDeviceToHost, slot.stream));
                slot.pending = true;
            }
            dm.h2d_bytes += n * sizeof(T);
            dm.d2h_bytes += sizeof(T);
            ++dm.chunks;
        }
        for (auto& slot : pipe.slots) {
            if (slot.pending) {
                CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                merged = combine(merged, *slot.h_output, cfg.operation);
                slot.pending = false;
            }
        }
        final = merged;
    }
    dm.elements = range.count * static_cast<std::size_t>(cfg.reuse_count);
    return final;
}

template <typename T>
Value reduce_cpu_file_range(const WorkerConfig& cfg, Range range, CpuMetrics& cm) {
    if (range.count == 0) return Value::identity(cfg.dtype, cfg.operation);
    const std::size_t capacity = std::min(range.count, std::max<std::size_t>(1, cfg.chunk_size));
    std::vector<T> buffer(capacity);
    Value final = Value::identity(cfg.dtype, cfg.operation);
    for (int reuse = 0; reuse < cfg.reuse_count; ++reuse) {
        std::ifstream input(cfg.dataset_path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open dataset for CPU file stream");
        input.seekg(static_cast<std::streamoff>(range.offset * sizeof(T)), std::ios::beg);
        if (!input) throw std::runtime_error("cannot seek CPU file stream");
        Value merged = Value::identity(cfg.dtype, cfg.operation);
        for (std::size_t done = 0; done < range.count; done += capacity) {
            const std::size_t n = std::min(capacity, range.count - done);
            cm.storage_read_us += timed_us([&] { read_exact(input, buffer.data(), n); });
            cm.storage_read_bytes += n * sizeof(T);
            auto partial = reduce_cpu(buffer.data(), n, cfg.dtype, cfg.cpu_backend, cfg.cpu_threads, cfg.operation);
            cm.compute_us += partial.compute_us;
            merged.combine(partial.result, cfg.dtype, cfg.operation);
            ++cm.chunks;
        }
        final = merged;
    }
    cm.elements = range.count * static_cast<std::size_t>(cfg.reuse_count);
    return final;
}

template <typename T>
class FileStreamStrategy final : public IntegratedBase {
public:
    explicit FileStreamStrategy(const WorkerConfig& cfg) : cfg_(cfg) {}

    void prepare(int warmup_runs) override {
        const std::size_t gpu_count = cfg_.gpu_ids.size();
        std::size_t cpu_count = 0;
        if (cfg_.scheduler == SchedulerKind::CpuOnly) {
            cpu_count = cfg_.count;
        } else if (cfg_.scheduler == SchedulerKind::StaticEqual) {
            if (cfg_.cpu_fraction >= 0.0) {
                cpu_count = std::min(
                    cfg_.count,
                    static_cast<std::size_t>(static_cast<double>(cfg_.count) * cfg_.cpu_fraction));
            } else {
                cpu_count = cfg_.count / (gpu_count + 1);
            }
        }
        cpu_range_ = {0, cpu_count};
        gpu_ranges_ = partition(cpu_count, cfg_.count - cpu_count, gpu_count);

        prepare_metrics_.partition.clear();
        if (cpu_count) prepare_metrics_.partition.push_back({"cpu", 0, -1, 0, cpu_count});
        for (std::size_t i = 0; i < gpu_count; ++i) {
            prepare_metrics_.partition.push_back({
                "gpu", static_cast<int>(i), cfg_.gpu_ids[i], gpu_ranges_[i].offset, gpu_ranges_[i].count
            });
        }

        const std::size_t chunk = cfg_.pipeline_chunk_elements > 0
            ? cfg_.pipeline_chunk_elements
            : cfg_.chunk_size;
        const int stream_count = cfg_.transfer_policy == TransferPolicy::AsyncPipeline
            ? cfg_.pipeline_streams : 1;
        for (std::size_t i = 0; i < gpu_count; ++i) {
            const int node = i < cfg_.gpu_numa_nodes.size() ? cfg_.gpu_numa_nodes[i] : -1;
            const std::size_t cap = std::max<std::size_t>(1, std::min(chunk, std::max<std::size_t>(1, gpu_ranges_[i].count)));
            pipes_.push_back(std::make_unique<GpuFilePipeline<T>>(
                cfg_.gpu_ids[i], node, cap, stream_count, cfg_.operation));
        }
        run_warmups(warmup_runs);
    }

    IterationMetrics run_once() override {
        const auto begin = Clock::now();
        IterationMetrics out;
        out.result = Value::identity(cfg_.dtype, cfg_.operation);
        out.gpus.resize(cfg_.gpu_ids.size());
        std::vector<T> gpu_values(cfg_.gpu_ids.size(), identity<T>(cfg_.operation));
        std::vector<std::exception_ptr> errors(cfg_.gpu_ids.size());
        std::vector<std::thread> threads;
        threads.reserve(cfg_.gpu_ids.size());

        for (std::size_t i = 0; i < cfg_.gpu_ids.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    if (i < cfg_.gpu_worker_cpus.size()) pin_current_thread(cfg_.gpu_worker_cpus[i]);
                    const auto gpu_start = Clock::now();
                    gpu_values[i] = reduce_gpu_file_range(cfg_, gpu_ranges_[i], *pipes_[i], out.gpus[i]);
                    out.gpus[i].total_us = std::chrono::duration<double, std::micro>(Clock::now() - gpu_start).count();
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }

        Value cpu_value = Value::identity(cfg_.dtype, cfg_.operation);
        if (cpu_range_.count) cpu_value = reduce_cpu_file_range<T>(cfg_, cpu_range_, out.cpu);

        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);

        const auto merge_begin = Clock::now();
        out.result = cpu_value;
        for (const auto& value : gpu_values) out.result.combine(to_value(value), cfg_.dtype, cfg_.operation);
        const auto merge_end = Clock::now();
        out.merge_us = std::chrono::duration<double, std::micro>(merge_end - merge_begin).count();
        out.e2e_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        return out;
    }

private:
    WorkerConfig cfg_;
    Range cpu_range_;
    std::vector<Range> gpu_ranges_;
    std::vector<std::unique_ptr<GpuFilePipeline<T>>> pipes_;
};

template <typename Strategy, typename... Args>
std::unique_ptr<IReductionStrategy> by_dtype(DataType dtype, Args&&... args) {
    switch (dtype) {
        case DataType::Int32: return std::make_unique<Strategy>(std::forward<Args>(args)...);
        default: break;
    }
    return nullptr;
}

template <template <typename> class Strategy>
std::unique_ptr<IReductionStrategy> make_typed(const WorkerConfig& cfg) {
    switch (cfg.dtype) {
        case DataType::Int32: return std::make_unique<Strategy<std::int32_t>>(cfg);
        case DataType::Int64: return std::make_unique<Strategy<std::int64_t>>(cfg);
        case DataType::Float32: return std::make_unique<Strategy<float>>(cfg);
        case DataType::Float64: return std::make_unique<Strategy<double>>(cfg);
    }
    throw std::logic_error("unreachable dtype");
}

}  // namespace

bool uses_integrated_strategy(const WorkerConfig& config) noexcept {
    if (config.storage_policy != "host_resident") return true;
    if (config.scheduler != SchedulerKind::GpuOnly || config.gpu_backend != GpuBackendKind::Cub) return false;
    if (config.memory_path == "chunked_sync" || config.memory_path == "pinned_direct" ||
        config.memory_path == "zero_copy" || config.memory_path == "managed_fault" ||
        config.memory_path == "managed_prefetch" || config.memory_path == "managed_advised" ||
        config.memory_path == "hmm_system") return true;
    if (config.reuse_count != 1 || config.use_cuda_graphs) return true;
    return false;
}

std::string integrated_unsupported_reason(const WorkerConfig& config) {
    if (config.storage_policy == "gds") {
        return "GPUDirect Storage unavailable in the unified worker on this build; cuFile/GDS support is required";
    }
    if (config.storage_policy == "file_stream") {
        const bool supported_scheduler =
            config.scheduler == SchedulerKind::CpuOnly ||
            config.scheduler == SchedulerKind::GpuOnly ||
            config.scheduler == SchedulerKind::GpuStaticEqual ||
            config.scheduler == SchedulerKind::StaticEqual;
        if (!supported_scheduler) {
            return "file_stream currently supports cpu_only, gpu_only, gpu_static_equal and static_equal schedulers";
        }
        return {};
    }
    if (!uses_integrated_strategy(config)) return {};
    if (config.gpu_ids.size() != 1) {
        return "host-resident special memory paths currently require exactly one GPU; use multi-GPU async topology algorithms for 2+ GPUs";
    }
    if (config.memory_path == "hmm_system") {
        int pageable = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, config.gpu_ids.front()));
        if (!pageable) return "HMM/system pageable GPU access unavailable: cudaDevAttrPageableMemoryAccess=0";
    }
    if (config.memory_path == "zero_copy") {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, config.gpu_ids.front()));
        if (!prop.canMapHostMemory) return "zero-copy unavailable: canMapHostMemory=0";
    }
    if (config.memory_path.rfind("managed_", 0) == 0) {
        int managed = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, config.gpu_ids.front()));
        if (!managed) return "Unified Memory unavailable: cudaDevAttrManagedMemory=0";
    }
    return {};
}

std::unique_ptr<IReductionStrategy> make_integrated_strategy(const WorkerConfig& config) {
    if (config.storage_policy == "file_stream") return make_typed<FileStreamStrategy>(config);
    if (config.storage_policy == "gds") {
        throw std::runtime_error("GDS strategy requested without integrated cuFile backend");
    }
    return make_typed<SingleGpuMemoryStrategy>(config);
}

}  // namespace prbench
