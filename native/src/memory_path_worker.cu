#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(_e)); \
    } \
} while (0)

enum class DType { Int32, Int64, Float32, Float64 };
enum class Op { Sum, Min, Max };
enum class Mode {
    ExplicitSync,
    ChunkedSync,
    AsyncPipeline,
    ZeroCopy,
    ManagedFault,
    ManagedPrefetch,
    ManagedAdvised,
    DeviceResident,
    HmmSystem
};

struct Config {
    std::filesystem::path dataset;
    DType dtype{DType::Float32};
    Op op{Op::Sum};
    Mode mode{Mode::ExplicitSync};
    std::size_t count{0};
    std::size_t chunk_elements{16u << 20};
    int streams{4};
    int device{0};
    int reuse_count{1};
    int warmup{1};
    int repetitions{3};
    bool use_cuda_graphs{false};
};

struct Sample {
    double total_ms{0.0};
    double h2d_ms{0.0};
    double kernel_ms{0.0};
    double d2h_ms{0.0};
    std::uint64_t host_to_device_bytes{0};
    std::uint64_t device_to_host_bytes{0};
    std::uint64_t remote_host_read_bytes{0};
    std::size_t chunks{0};
};

struct Features {
    int managed{0};
    int concurrent_managed{0};
    int pageable{0};
    int host_tables{0};
    int map_host{0};
};

std::string mode_name(Mode mode) {
    switch (mode) {
        case Mode::ExplicitSync: return "explicit_sync";
        case Mode::ChunkedSync: return "chunked_sync";
        case Mode::AsyncPipeline: return "async_pipeline";
        case Mode::ZeroCopy: return "zero_copy";
        case Mode::ManagedFault: return "managed_fault";
        case Mode::ManagedPrefetch: return "managed_prefetch";
        case Mode::ManagedAdvised: return "managed_advised";
        case Mode::DeviceResident: return "device_resident";
        case Mode::HmmSystem: return "hmm_system";
    }
    return "unknown";
}

DType parse_dtype(std::string_view s) {
    if (s == "int32") return DType::Int32;
    if (s == "int64") return DType::Int64;
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    throw std::invalid_argument("unsupported dtype: " + std::string(s));
}

Op parse_op(std::string_view s) {
    if (s == "sum") return Op::Sum;
    if (s == "min") return Op::Min;
    if (s == "max") return Op::Max;
    throw std::invalid_argument("unsupported operation: " + std::string(s));
}

Mode parse_mode(std::string_view s) {
    if (s == "explicit_sync") return Mode::ExplicitSync;
    if (s == "chunked_sync") return Mode::ChunkedSync;
    if (s == "async_pipeline") return Mode::AsyncPipeline;
    if (s == "zero_copy") return Mode::ZeroCopy;
    if (s == "managed_fault") return Mode::ManagedFault;
    if (s == "managed_prefetch") return Mode::ManagedPrefetch;
    if (s == "managed_advised") return Mode::ManagedAdvised;
    if (s == "device_resident") return Mode::DeviceResident;
    if (s == "hmm_system") return Mode::HmmSystem;
    throw std::invalid_argument("unsupported mode: " + std::string(s));
}

Config parse_cli(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--dataset") c.dataset = next();
        else if (arg == "--dtype") c.dtype = parse_dtype(next());
        else if (arg == "--operation") c.op = parse_op(next());
        else if (arg == "--mode") c.mode = parse_mode(next());
        else if (arg == "--count") c.count = std::stoull(next());
        else if (arg == "--chunk-elements") c.chunk_elements = std::stoull(next());
        else if (arg == "--streams") c.streams = std::stoi(next());
        else if (arg == "--device") c.device = std::stoi(next());
        else if (arg == "--reuse-count") c.reuse_count = std::stoi(next());
        else if (arg == "--warmup") c.warmup = std::stoi(next());
        else if (arg == "--repetitions") c.repetitions = std::stoi(next());
        else if (arg == "--cuda-graphs") c.use_cuda_graphs = true;
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (c.dataset.empty() || c.count == 0) {
        throw std::invalid_argument("--dataset and --count are required");
    }
    if (c.chunk_elements == 0 || c.streams < 1 || c.reuse_count < 1 || c.warmup < 0 || c.repetitions < 1) {
        throw std::invalid_argument("invalid numeric option");
    }
    if (c.use_cuda_graphs && c.mode != Mode::ExplicitSync && c.mode != Mode::DeviceResident) {
        throw std::invalid_argument("CUDA graphs are currently supported only for explicit_sync/device_resident");
    }
    return c;
}

Features query_features(int device) {
    Features f;
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    f.map_host = p.canMapHostMemory;
    CUDA_CHECK(cudaDeviceGetAttribute(&f.managed, cudaDevAttrManagedMemory, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&f.concurrent_managed, cudaDevAttrConcurrentManagedAccess, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&f.pageable, cudaDevAttrPageableMemoryAccess, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&f.host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
    return f;
}

template<class T>
T identity(Op op) {
    if (op == Op::Sum) return T{0};
    if (op == Op::Min) return std::numeric_limits<T>::max();
    return std::numeric_limits<T>::lowest();
}

template<class T>
T combine(T a, T b, Op op) {
    if (op == Op::Sum) return a + b;
    if (op == Op::Min) return std::min(a, b);
    return std::max(a, b);
}

template<class T>
void read_exact(const Config& c, T* dst) {
    std::ifstream input(c.dataset, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset: " + c.dataset.string());
    input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(c.count * sizeof(T)));
    if (!input) throw std::runtime_error("cannot read complete dataset");
}

template<class T>
std::vector<T> read_pageable(const Config& c) {
    std::vector<T> data(c.count);
    read_exact(c, data.data());
    return data;
}

template<class T>
void cub_query(const T* input, T* output, std::size_t n, Op op, cudaStream_t stream,
               void*& temp, std::size_t& temp_bytes) {
    temp_bytes = 0;
    if (op == Op::Sum) CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, output, n, stream));
    else if (op == Op::Min) CUDA_CHECK(cub::DeviceReduce::Min(nullptr, temp_bytes, input, output, n, stream));
    else CUDA_CHECK(cub::DeviceReduce::Max(nullptr, temp_bytes, input, output, n, stream));
    if (temp_bytes > 0) CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
}

template<class T>
void cub_reduce(void* temp, std::size_t temp_bytes, const T* input, T* output,
                std::size_t n, Op op, cudaStream_t stream) {
    if (op == Op::Sum) CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, input, output, n, stream));
    else if (op == Op::Min) CUDA_CHECK(cub::DeviceReduce::Min(temp, temp_bytes, input, output, n, stream));
    else CUDA_CHECK(cub::DeviceReduce::Max(temp, temp_bytes, input, output, n, stream));
}

template<class F>
double timed_ms(F&& f) {
    const auto start = Clock::now();
    f();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template<class T>
std::vector<Sample> run_full_device(const Config& c, const T* host, T& result) {
    T* d_input = nullptr;
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&d_input, c.count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    cub_query(d_input, d_output, c.count, c.op, stream, temp, temp_bytes);

    if (c.mode == Mode::DeviceResident) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, host, c.count * sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};
    if (c.use_cuda_graphs) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < c.reuse_count; ++k) {
            cub_reduce(temp, temp_bytes, d_input, d_output, c.count, c.op, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    }

    auto one = [&](bool measured) -> Sample {
        Sample s;
        if (c.mode == Mode::ExplicitSync) {
            s.h2d_ms = timed_ms([&] {
                CUDA_CHECK(cudaMemcpyAsync(d_input, host, c.count * sizeof(T), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            s.host_to_device_bytes = c.count * sizeof(T);
        }
        const auto total_start = Clock::now();
        if (c.use_cuda_graphs) {
            s.kernel_ms = timed_ms([&] {
                CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            result = *h_output;
            s.device_to_host_bytes = sizeof(T);
        } else {
            s.kernel_ms = timed_ms([&] {
                for (int k = 0; k < c.reuse_count; ++k) {
                    cub_reduce(temp, temp_bytes, d_input, d_output, c.count, c.op, stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            s.d2h_ms = timed_ms([&] {
                CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            result = *h_output;
            s.device_to_host_bytes = sizeof(T);
        }
        s.total_ms = s.h2d_ms + std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        s.chunks = 1;
        (void)measured;
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one(false);
    std::vector<Sample> samples;
    samples.reserve(c.repetitions);
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one(true));

    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    if (temp) cudaFree(temp);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);
    return samples;
}

template<class T>
std::vector<Sample> run_chunked_sync(const Config& c, const T* host, T& result) {
    const std::size_t capacity = std::min(c.count, c.chunk_elements);
    T* d_input = nullptr;
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&d_input, capacity * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    cub_query(d_input, d_output, capacity, c.op, stream, temp, temp_bytes);

    auto one = [&]() -> Sample {
        Sample s;
        const auto total_start = Clock::now();
        T final_result = identity<T>(c.op);
        for (int reuse = 0; reuse < c.reuse_count; ++reuse) {
            T merged = identity<T>(c.op);
            for (std::size_t offset = 0; offset < c.count; offset += capacity) {
                const std::size_t n = std::min(capacity, c.count - offset);
                s.h2d_ms += timed_ms([&] {
                    CUDA_CHECK(cudaMemcpyAsync(d_input, host + offset, n * sizeof(T), cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                });
                s.host_to_device_bytes += n * sizeof(T);
                s.kernel_ms += timed_ms([&] {
                    cub_reduce(temp, temp_bytes, d_input, d_output, n, c.op, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                });
                s.d2h_ms += timed_ms([&] {
                    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                });
                s.device_to_host_bytes += sizeof(T);
                merged = combine(merged, *h_output, c.op);
                ++s.chunks;
            }
            final_result = merged;
        }
        result = final_result;
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    samples.reserve(c.repetitions);
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());

    if (temp) cudaFree(temp);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);
    return samples;
}

template<class T>
std::vector<Sample> run_async_pipeline(const Config& c, const T* host, T& result) {
    const std::size_t capacity = std::min(c.count, c.chunk_elements);
    struct Slot {
        cudaStream_t stream{};
        T* staging{nullptr};
        T* d_input{nullptr};
        T* d_output{nullptr};
        T* h_output{nullptr};
        void* temp{nullptr};
        std::size_t temp_bytes{0};
        bool pending{false};
    };
    std::vector<Slot> slots(static_cast<std::size_t>(c.streams));
    for (auto& slot : slots) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaHostAlloc(&slot.staging, capacity * sizeof(T), cudaHostAllocPortable));
        CUDA_CHECK(cudaMalloc(&slot.d_input, capacity * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&slot.d_output, sizeof(T)));
        CUDA_CHECK(cudaHostAlloc(&slot.h_output, sizeof(T), cudaHostAllocPortable));
        cub_query(slot.d_input, slot.d_output, capacity, c.op, slot.stream, slot.temp, slot.temp_bytes);
    }

    auto one = [&]() -> Sample {
        Sample s;
        const auto total_start = Clock::now();
        T final_result = identity<T>(c.op);
        for (int reuse = 0; reuse < c.reuse_count; ++reuse) {
            T merged = identity<T>(c.op);
            std::size_t chunk_index = 0;
            for (std::size_t offset = 0; offset < c.count; offset += capacity, ++chunk_index) {
                Slot& slot = slots[chunk_index % slots.size()];
                if (slot.pending) {
                    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                    merged = combine(merged, *slot.h_output, c.op);
                    slot.pending = false;
                }
                const std::size_t n = std::min(capacity, c.count - offset);
                std::memcpy(slot.staging, host + offset, n * sizeof(T));
                CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.staging, n * sizeof(T), cudaMemcpyHostToDevice, slot.stream));
                cub_reduce(slot.temp, slot.temp_bytes, slot.d_input, slot.d_output, n, c.op, slot.stream);
                CUDA_CHECK(cudaMemcpyAsync(slot.h_output, slot.d_output, sizeof(T), cudaMemcpyDeviceToHost, slot.stream));
                slot.pending = true;
                s.host_to_device_bytes += n * sizeof(T);
                s.device_to_host_bytes += sizeof(T);
                ++s.chunks;
            }
            for (auto& slot : slots) {
                if (slot.pending) {
                    CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                    merged = combine(merged, *slot.h_output, c.op);
                    slot.pending = false;
                }
            }
            final_result = merged;
        }
        result = final_result;
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    samples.reserve(c.repetitions);
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());

    for (auto& slot : slots) {
        if (slot.temp) cudaFree(slot.temp);
        cudaFree(slot.d_input);
        cudaFree(slot.d_output);
        cudaFreeHost(slot.h_output);
        cudaFreeHost(slot.staging);
        cudaStreamDestroy(slot.stream);
    }
    return samples;
}

template<class T>
std::vector<Sample> run_zero_copy(const Config& c, T& result) {
    const auto f = query_features(c.device);
    if (!f.map_host) throw std::runtime_error("zero-copy unavailable: canMapHostMemory=0");
    T* host = nullptr;
    T* mapped = nullptr;
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    CUDA_CHECK(cudaHostAlloc(&host, c.count * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    read_exact(c, host);
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), host, 0));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cub_query(mapped, d_output, c.count, c.op, stream, temp, temp_bytes);

    auto one = [&]() -> Sample {
        Sample s;
        const auto total_start = Clock::now();
        s.kernel_ms = timed_ms([&] {
            for (int k = 0; k < c.reuse_count; ++k) {
                cub_reduce(temp, temp_bytes, mapped, d_output, c.count, c.op, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        s.d2h_ms = timed_ms([&] {
            CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        result = *h_output;
        s.remote_host_read_bytes = static_cast<std::uint64_t>(c.count * sizeof(T)) * c.reuse_count;
        s.device_to_host_bytes = sizeof(T);
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        s.chunks = 1;
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    samples.reserve(c.repetitions);
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());

    if (temp) cudaFree(temp);
    cudaFree(d_output);
    cudaFreeHost(h_output);
    cudaFreeHost(host);
    cudaStreamDestroy(stream);
    return samples;
}

template<class T>
std::vector<Sample> run_managed(const Config& c, T& result) {
    const auto f = query_features(c.device);
    if (!f.managed) throw std::runtime_error("managed memory unavailable: cudaDevAttrManagedMemory=0");
    T* data = nullptr;
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    const std::size_t bytes = c.count * sizeof(T);
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    read_exact(c, data);
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (c.mode == Mode::ManagedAdvised) {
        CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, c.device));
        CUDA_CHECK(cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, c.device));
    }
    cub_query(data, d_output, c.count, c.op, stream, temp, temp_bytes);

    auto reset_host_residency = [&] {
        if (f.concurrent_managed) {
            CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            // CPU touch is outside the measurement window. On older managed-memory systems
            // it establishes the host as the most recent accessor without changing values.
            volatile T sink{};
            const std::size_t stride = std::max<std::size_t>(1, 4096 / sizeof(T));
            for (std::size_t i = 0; i < c.count; i += stride) sink = combine(sink, data[i], Op::Sum);
            (void)sink;
        }
    };

    auto one = [&]() -> Sample {
        reset_host_residency();
        Sample s;
        const auto total_start = Clock::now();
        if (c.mode == Mode::ManagedPrefetch) {
            s.h2d_ms = timed_ms([&] {
                CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, c.device, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            s.host_to_device_bytes = bytes;
        }
        s.kernel_ms = timed_ms([&] {
            for (int k = 0; k < c.reuse_count; ++k) {
                cub_reduce(temp, temp_bytes, data, d_output, c.count, c.op, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        s.d2h_ms = timed_ms([&] {
            CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        result = *h_output;
        s.device_to_host_bytes = sizeof(T);
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        s.chunks = 1;
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    samples.reserve(c.repetitions);
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());

    if (temp) cudaFree(temp);
    cudaFree(d_output);
    cudaFreeHost(h_output);
    cudaFree(data);
    cudaStreamDestroy(stream);
    return samples;
}

template<class T>
std::vector<Sample> run_hmm(const Config& c, T& result) {
    const auto f = query_features(c.device);
    if (!f.pageable) {
        throw std::runtime_error("HMM/system pageable GPU access unavailable: cudaDevAttrPageableMemoryAccess=0");
    }
    std::unique_ptr<T[]> host(new T[c.count]);
    read_exact(c, host.get());
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cub_query(host.get(), d_output, c.count, c.op, stream, temp, temp_bytes);

    auto one = [&]() -> Sample {
        Sample s;
        const auto total_start = Clock::now();
        s.kernel_ms = timed_ms([&] {
            for (int k = 0; k < c.reuse_count; ++k) {
                cub_reduce(temp, temp_bytes, host.get(), d_output, c.count, c.op, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        s.d2h_ms = timed_ms([&] {
            CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        });
        result = *h_output;
        s.remote_host_read_bytes = static_cast<std::uint64_t>(c.count * sizeof(T)) * c.reuse_count;
        s.device_to_host_bytes = sizeof(T);
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        s.chunks = 1;
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    samples.reserve(c.repetitions);
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());

    if (temp) cudaFree(temp);
    cudaFree(d_output);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);
    return samples;
}

template<class T>
void execute(const Config& c) {
    CUDA_CHECK(cudaSetDevice(c.device));
    T result{};
    std::vector<Sample> samples;

    if (c.mode == Mode::ZeroCopy) {
        samples = run_zero_copy<T>(c, result);
    } else if (c.mode == Mode::ManagedFault || c.mode == Mode::ManagedPrefetch || c.mode == Mode::ManagedAdvised) {
        samples = run_managed<T>(c, result);
    } else if (c.mode == Mode::HmmSystem) {
        samples = run_hmm<T>(c, result);
    } else {
        auto host = read_pageable<T>(c);
        if (c.mode == Mode::ExplicitSync || c.mode == Mode::DeviceResident) {
            samples = run_full_device<T>(c, host.data(), result);
        } else if (c.mode == Mode::ChunkedSync) {
            samples = run_chunked_sync<T>(c, host.data(), result);
        } else if (c.mode == Mode::AsyncPipeline) {
            samples = run_async_pipeline<T>(c, host.data(), result);
        } else {
            throw std::logic_error("unreachable mode");
        }
    }

    auto mean_double = [&](auto member) {
        double total = 0.0;
        for (const auto& s : samples) total += member(s);
        return total / static_cast<double>(samples.size());
    };
    auto mean_u64 = [&](auto member) {
        std::uint64_t total = 0;
        for (const auto& s : samples) total += member(s);
        return total / samples.size();
    };
    auto mean_size = [&](auto member) {
        std::size_t total = 0;
        for (const auto& s : samples) total += member(s);
        return total / samples.size();
    };

    const auto feat = query_features(c.device);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, c.device));
    std::cout
        << "{\"event\":\"memory_path_result\""
        << ",\"mode\":\"" << mode_name(c.mode) << "\""
        << ",\"device\":" << c.device
        << ",\"gpu_name\":\"" << prop.name << "\""
        << ",\"count\":" << c.count
        << ",\"element_bytes\":" << sizeof(T)
        << ",\"reuse_count\":" << c.reuse_count
        << ",\"chunk_elements\":" << c.chunk_elements
        << ",\"streams\":" << c.streams
        << ",\"mean_total_ms\":" << mean_double([](const Sample& s) { return s.total_ms; })
        << ",\"mean_h2d_ms\":" << mean_double([](const Sample& s) { return s.h2d_ms; })
        << ",\"mean_kernel_ms\":" << mean_double([](const Sample& s) { return s.kernel_ms; })
        << ",\"mean_d2h_ms\":" << mean_double([](const Sample& s) { return s.d2h_ms; })
        << ",\"mean_h2d_bytes\":" << mean_u64([](const Sample& s) { return s.host_to_device_bytes; })
        << ",\"mean_d2h_bytes\":" << mean_u64([](const Sample& s) { return s.device_to_host_bytes; })
        << ",\"mean_remote_host_read_bytes\":" << mean_u64([](const Sample& s) { return s.remote_host_read_bytes; })
        << ",\"mean_chunks\":" << mean_size([](const Sample& s) { return s.chunks; })
        << ",\"cuda_managed_memory\":" << feat.managed
        << ",\"cuda_concurrent_managed_access\":" << feat.concurrent_managed
        << ",\"cuda_pageable_memory_access\":" << feat.pageable
        << ",\"cuda_pageable_uses_host_page_tables\":" << feat.host_tables
        << ",\"cuda_can_map_host_memory\":" << feat.map_host
        << ",\"cuda_graphs_used\":" << (c.use_cuda_graphs ? "true" : "false")
        << ",\"result\":" << result
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_cli(argc, argv);
        switch (config.dtype) {
            case DType::Int32: execute<std::int32_t>(config); break;
            case DType::Int64: execute<std::int64_t>(config); break;
            case DType::Float32: execute<float>(config); break;
            case DType::Float64: execute<double>(config); break;
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "memory-path-worker: " << exc.what() << "\n";
        return 2;
    }
}
