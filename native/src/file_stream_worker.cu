#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
enum class Mode { ChunkedSync, AsyncPipeline };

struct Config {
    std::filesystem::path dataset;
    DType dtype{DType::Float32};
    Op op{Op::Sum};
    Mode mode{Mode::ChunkedSync};
    std::size_t count{0};
    std::size_t chunk_elements{16u << 20};
    int streams{4};
    int device{0};
    int reuse_count{1};
    int warmup{0};
    int repetitions{1};
};

struct Sample {
    double total_ms{0.0};
    double storage_read_ms{0.0};
    double h2d_ms{0.0};
    double kernel_ms{0.0};
    double d2h_ms{0.0};
    std::uint64_t storage_read_bytes{0};
    std::uint64_t host_to_device_bytes{0};
    std::uint64_t device_to_host_bytes{0};
    std::size_t chunks{0};
};

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
    if (s == "chunked_sync") return Mode::ChunkedSync;
    if (s == "async_pipeline") return Mode::AsyncPipeline;
    throw std::invalid_argument("file-stream worker supports only chunked_sync/async_pipeline");
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
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (c.dataset.empty() || c.count == 0 || c.chunk_elements == 0 || c.streams < 1 ||
        c.reuse_count < 1 || c.warmup < 0 || c.repetitions < 1) {
        throw std::invalid_argument("invalid or missing option");
    }
    return c;
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
void read_chunk(std::ifstream& input, T* dst, std::size_t n) {
    input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n * sizeof(T)));
    if (input.gcount() != static_cast<std::streamsize>(n * sizeof(T))) {
        throw std::runtime_error("short read from dataset");
    }
}

template<class T>
std::vector<Sample> run_chunked(const Config& c, T& result) {
    const std::size_t capacity = std::min(c.count, c.chunk_elements);
    T* staging = nullptr;
    T* d_input = nullptr;
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    CUDA_CHECK(cudaHostAlloc(&staging, capacity * sizeof(T), cudaHostAllocPortable));
    CUDA_CHECK(cudaMalloc(&d_input, capacity * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cub_query(d_input, d_output, capacity, c.op, stream, temp, temp_bytes);

    auto one = [&]() {
        Sample s;
        const auto total_start = Clock::now();
        T final_result = identity<T>(c.op);
        for (int reuse = 0; reuse < c.reuse_count; ++reuse) {
            std::ifstream input(c.dataset, std::ios::binary);
            if (!input) throw std::runtime_error("cannot open dataset: " + c.dataset.string());
            T merged = identity<T>(c.op);
            for (std::size_t offset = 0; offset < c.count; offset += capacity) {
                const std::size_t n = std::min(capacity, c.count - offset);
                s.storage_read_ms += timed_ms([&] { read_chunk(input, staging, n); });
                s.storage_read_bytes += n * sizeof(T);
                s.h2d_ms += timed_ms([&] {
                    CUDA_CHECK(cudaMemcpyAsync(d_input, staging, n * sizeof(T), cudaMemcpyHostToDevice, stream));
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
    cudaFreeHost(staging);
    cudaStreamDestroy(stream);
    return samples;
}

template<class T>
std::vector<Sample> run_async(const Config& c, T& result) {
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

    auto one = [&]() {
        Sample s;
        const auto total_start = Clock::now();
        T final_result = identity<T>(c.op);
        for (int reuse = 0; reuse < c.reuse_count; ++reuse) {
            std::ifstream input(c.dataset, std::ios::binary);
            if (!input) throw std::runtime_error("cannot open dataset: " + c.dataset.string());
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
                s.storage_read_ms += timed_ms([&] { read_chunk(input, slot.staging, n); });
                s.storage_read_bytes += n * sizeof(T);
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
void execute(const Config& c) {
    CUDA_CHECK(cudaSetDevice(c.device));
    const auto expected_bytes = static_cast<std::uintmax_t>(c.count) * sizeof(T);
    if (!std::filesystem::exists(c.dataset) || std::filesystem::file_size(c.dataset) != expected_bytes) {
        throw std::runtime_error("dataset file size does not match --count/--dtype");
    }

    T result{};
    const auto samples = c.mode == Mode::ChunkedSync
        ? run_chunked<T>(c, result)
        : run_async<T>(c, result);

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

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, c.device));
    std::cout
        << "{\"event\":\"file_stream_result\""
        << ",\"storage_policy\":\"file_stream\""
        << ",\"mode\":\"" << (c.mode == Mode::ChunkedSync ? "chunked_sync" : "async_pipeline") << "\""
        << ",\"device\":" << c.device
        << ",\"gpu_name\":\"" << prop.name << "\""
        << ",\"count\":" << c.count
        << ",\"element_bytes\":" << sizeof(T)
        << ",\"reuse_count\":" << c.reuse_count
        << ",\"chunk_elements\":" << c.chunk_elements
        << ",\"streams\":" << c.streams
        << ",\"mean_total_ms\":" << mean_double([](const Sample& s) { return s.total_ms; })
        << ",\"mean_storage_read_ms\":" << mean_double([](const Sample& s) { return s.storage_read_ms; })
        << ",\"mean_h2d_ms\":" << mean_double([](const Sample& s) { return s.h2d_ms; })
        << ",\"mean_kernel_ms\":" << mean_double([](const Sample& s) { return s.kernel_ms; })
        << ",\"mean_d2h_ms\":" << mean_double([](const Sample& s) { return s.d2h_ms; })
        << ",\"mean_storage_read_bytes\":" << mean_u64([](const Sample& s) { return s.storage_read_bytes; })
        << ",\"mean_h2d_bytes\":" << mean_u64([](const Sample& s) { return s.host_to_device_bytes; })
        << ",\"mean_d2h_bytes\":" << mean_u64([](const Sample& s) { return s.device_to_host_bytes; })
        << ",\"mean_chunks\":" << mean_size([](const Sample& s) { return s.chunks; })
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
        std::cerr << "file-stream-worker: " << exc.what() << "\n";
        return 2;
    }
}
