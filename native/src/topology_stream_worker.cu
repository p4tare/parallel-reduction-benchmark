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
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if PRBENCH_HAS_LIBNUMA
#include <numa.h>
#endif

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
enum class Mode { PinnedDirect, MultiGpuAsync, HybridCpuGpu };
enum class StoragePolicy { HostResident, FileStream };

struct Config {
    std::filesystem::path dataset;
    DType dtype{DType::Float32};
    Op op{Op::Sum};
    Mode mode{Mode::MultiGpuAsync};
    StoragePolicy storage{StoragePolicy::HostResident};
    std::size_t count{0};
    std::size_t chunk_elements{16u << 20};
    int streams{2};
    std::vector<int> gpu_ids{0};
    std::vector<int> gpu_numa_nodes;
    int cpu_threads{1};
    double cpu_fraction{0.25};
    int warmup{1};
    int repetitions{3};
    bool numa_strict{false};
};

struct Range {
    std::size_t offset{0};
    std::size_t count{0};
};

struct Sample {
    double total_ms{0.0};
    double storage_read_ms{0.0};
    double host_memcpy_ms{0.0};
    double h2d_ms{0.0};
    double kernel_ms{0.0};
    double d2h_ms{0.0};
    double cpu_ms{0.0};
    std::uint64_t storage_read_bytes{0};
    std::uint64_t h2d_bytes{0};
    std::uint64_t d2h_bytes{0};
    std::size_t chunks{0};
    std::size_t cpu_elements{0};
    bool numa_applied{false};
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
    if (s == "pinned_direct") return Mode::PinnedDirect;
    if (s == "multi_gpu_async") return Mode::MultiGpuAsync;
    if (s == "hybrid_cpu_gpu") return Mode::HybridCpuGpu;
    throw std::invalid_argument("unsupported topology-stream mode: " + std::string(s));
}

StoragePolicy parse_storage(std::string_view s) {
    if (s == "host_resident") return StoragePolicy::HostResident;
    if (s == "file_stream") return StoragePolicy::FileStream;
    throw std::invalid_argument("unsupported storage policy: " + std::string(s));
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    if (out.empty()) throw std::invalid_argument("integer list cannot be empty");
    return out;
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
        else if (arg == "--storage-policy") c.storage = parse_storage(next());
        else if (arg == "--count") c.count = std::stoull(next());
        else if (arg == "--chunk-elements") c.chunk_elements = std::stoull(next());
        else if (arg == "--streams") c.streams = std::stoi(next());
        else if (arg == "--gpu-ids") c.gpu_ids = parse_int_list(next());
        else if (arg == "--gpu-numa-nodes") c.gpu_numa_nodes = parse_int_list(next());
        else if (arg == "--cpu-threads") c.cpu_threads = std::stoi(next());
        else if (arg == "--cpu-fraction") c.cpu_fraction = std::stod(next());
        else if (arg == "--warmup") c.warmup = std::stoi(next());
        else if (arg == "--repetitions") c.repetitions = std::stoi(next());
        else if (arg == "--numa-strict") c.numa_strict = true;
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (c.dataset.empty() || c.count == 0 || c.chunk_elements == 0 || c.streams < 1 ||
        c.cpu_threads < 1 || c.warmup < 0 || c.repetitions < 1) {
        throw std::invalid_argument("invalid or missing numeric option");
    }
    if (c.cpu_fraction < 0.0 || c.cpu_fraction >= 1.0) {
        throw std::invalid_argument("--cpu-fraction must be in [0,1)");
    }
    if (!c.gpu_numa_nodes.empty() && c.gpu_numa_nodes.size() != c.gpu_ids.size()) {
        throw std::invalid_argument("--gpu-numa-nodes must match --gpu-ids length");
    }
    if (c.mode == Mode::PinnedDirect && c.storage != StoragePolicy::HostResident) {
        throw std::invalid_argument("pinned_direct requires host_resident storage");
    }
    if (c.mode == Mode::PinnedDirect && c.gpu_ids.size() != 1) {
        throw std::invalid_argument("pinned_direct requires exactly one GPU");
    }
    if ((c.mode == Mode::MultiGpuAsync || c.mode == Mode::HybridCpuGpu) && c.gpu_ids.empty()) {
        throw std::invalid_argument("streaming topology requires at least one GPU");
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

template<class F>
double timed_ms(F&& f) {
    const auto begin = Clock::now();
    f();
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

std::vector<Range> equal_partition(std::size_t offset, std::size_t count, std::size_t parts) {
    std::vector<Range> out(parts);
    const std::size_t base = parts ? count / parts : 0;
    const std::size_t rem = parts ? count % parts : 0;
    std::size_t pos = offset;
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t n = base + (i < rem ? 1 : 0);
        out[i] = {pos, n};
        pos += n;
    }
    return out;
}

template<class T>
void cub_query(const T* input, T* output, std::size_t n, Op op, cudaStream_t stream,
               void*& temp, std::size_t& temp_bytes) {
    temp_bytes = 0;
    if (op == Op::Sum) CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, input, output, n, stream));
    else if (op == Op::Min) CUDA_CHECK(cub::DeviceReduce::Min(nullptr, temp_bytes, input, output, n, stream));
    else CUDA_CHECK(cub::DeviceReduce::Max(nullptr, temp_bytes, input, output, n, stream));
    if (temp_bytes) CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
}

template<class T>
void cub_reduce(void* temp, std::size_t temp_bytes, const T* input, T* output,
                std::size_t n, Op op, cudaStream_t stream) {
    if (op == Op::Sum) CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, input, output, n, stream));
    else if (op == Op::Min) CUDA_CHECK(cub::DeviceReduce::Min(temp, temp_bytes, input, output, n, stream));
    else CUDA_CHECK(cub::DeviceReduce::Max(temp, temp_bytes, input, output, n, stream));
}

struct HostBuffer {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool via_numa{false};
    bool registered{false};

    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&& other) noexcept {
        ptr = other.ptr; bytes = other.bytes; via_numa = other.via_numa; registered = other.registered;
        other.ptr = nullptr; other.bytes = 0; other.registered = false;
    }
    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr; bytes = other.bytes; via_numa = other.via_numa; registered = other.registered;
            other.ptr = nullptr; other.bytes = 0; other.registered = false;
        }
        return *this;
    }
    ~HostBuffer() { release(); }

    void release() {
        if (!ptr) return;
        if (registered) cudaHostUnregister(ptr);
#if PRBENCH_HAS_LIBNUMA
        if (via_numa) numa_free(ptr, bytes);
        else std::free(ptr);
#else
        std::free(ptr);
#endif
        ptr = nullptr;
    }
};

HostBuffer allocate_registered(std::size_t bytes, int numa_node, bool strict, bool& numa_applied) {
    HostBuffer out;
    out.bytes = bytes;
    constexpr std::size_t alignment = 4096;
#if PRBENCH_HAS_LIBNUMA
    if (numa_node >= 0 && numa_available() >= 0) {
        out.ptr = numa_alloc_onnode(bytes, numa_node);
        if (!out.ptr) throw std::runtime_error("numa_alloc_onnode failed");
        out.via_numa = true;
        numa_applied = true;
    } else
#endif
    {
        if (strict && numa_node >= 0) {
            throw std::runtime_error("NUMA-local staging unavailable: libnuma not available");
        }
        if (posix_memalign(&out.ptr, alignment, bytes) != 0 || !out.ptr) {
            throw std::runtime_error("posix_memalign failed for host staging");
        }
    }
    CUDA_CHECK(cudaHostRegister(out.ptr, bytes, cudaHostRegisterPortable));
    out.registered = true;
    return out;
}

template<class T>
void read_exact_at(const Config& c, std::size_t element_offset, T* dst, std::size_t n) {
    std::ifstream input(c.dataset, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset: " + c.dataset.string());
    input.seekg(static_cast<std::streamoff>(element_offset * sizeof(T)), std::ios::beg);
    if (!input) throw std::runtime_error("seek failed");
    input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n * sizeof(T)));
    if (input.gcount() != static_cast<std::streamsize>(n * sizeof(T))) {
        throw std::runtime_error("short read from dataset");
    }
}

template<class T>
std::vector<T> read_whole(const Config& c) {
    std::vector<T> data(c.count);
    read_exact_at(c, 0, data.data(), c.count);
    return data;
}

template<class T>
T cpu_reduce_memory(const T* input, std::size_t n, Op op, int threads) {
    if (n == 0) return identity<T>(op);
    const int actual = std::max(1, std::min<int>(threads, static_cast<int>(n)));
    std::vector<T> partials(static_cast<std::size_t>(actual), identity<T>(op));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(actual));
    for (int tid = 0; tid < actual; ++tid) {
        workers.emplace_back([&, tid] {
            const std::size_t begin = n * static_cast<std::size_t>(tid) / static_cast<std::size_t>(actual);
            const std::size_t end = n * static_cast<std::size_t>(tid + 1) / static_cast<std::size_t>(actual);
            T value = identity<T>(op);
            for (std::size_t i = begin; i < end; ++i) value = combine(value, input[i], op);
            partials[static_cast<std::size_t>(tid)] = value;
        });
    }
    for (auto& t : workers) t.join();
    T value = identity<T>(op);
    for (const auto& p : partials) value = combine(value, p, op);
    return value;
}

template<class T>
T cpu_reduce_file(const Config& c, Range range, int threads, std::size_t chunk_elements,
                  Sample& s) {
    if (range.count == 0) return identity<T>(c.op);
    const std::size_t capacity = std::min(range.count, chunk_elements);
    std::vector<T> buffer(capacity);
    std::ifstream input(c.dataset, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset for CPU stream");
    input.seekg(static_cast<std::streamoff>(range.offset * sizeof(T)), std::ios::beg);
    T merged = identity<T>(c.op);
    const auto cpu_start = Clock::now();
    for (std::size_t done = 0; done < range.count; done += capacity) {
        const std::size_t n = std::min(capacity, range.count - done);
        s.storage_read_ms += timed_ms([&] {
            input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(n * sizeof(T)));
            if (input.gcount() != static_cast<std::streamsize>(n * sizeof(T))) {
                throw std::runtime_error("short CPU stream read");
            }
        });
        s.storage_read_bytes += n * sizeof(T);
        merged = combine(merged, cpu_reduce_memory(buffer.data(), n, c.op, threads), c.op);
        ++s.chunks;
    }
    s.cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - cpu_start).count();
    s.cpu_elements += range.count;
    return merged;
}

template<class T>
struct GpuPipeline {
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
    Op op{Op::Sum};
    std::vector<Slot> slots;
    bool numa_applied{false};

    GpuPipeline(int dev, int node, std::size_t cap, int stream_count, Op operation, bool strict)
        : device(dev), numa_node(node), capacity(cap), op(operation), slots(static_cast<std::size_t>(stream_count)) {
        CUDA_CHECK(cudaSetDevice(device));
        for (auto& slot : slots) {
            slot.staging = allocate_registered(capacity * sizeof(T), numa_node, strict, numa_applied);
            CUDA_CHECK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&slot.d_input, capacity * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&slot.d_output, sizeof(T)));
            CUDA_CHECK(cudaHostAlloc(&slot.h_output, sizeof(T), cudaHostAllocPortable));
            cub_query(slot.d_input, slot.d_output, capacity, op, slot.stream, slot.temp, slot.temp_bytes);
        }
    }

    ~GpuPipeline() {
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

template<class T>
T reduce_gpu_range(const Config& c, Range range, const T* host_base, int gpu_index,
                   GpuPipeline<T>& pipe, Sample& s, std::mutex& sample_mutex) {
    CUDA_CHECK(cudaSetDevice(pipe.device));
#if PRBENCH_HAS_LIBNUMA
    if (pipe.numa_node >= 0 && numa_available() >= 0) {
        if (numa_run_on_node(pipe.numa_node) != 0 && c.numa_strict) {
            throw std::runtime_error("numa_run_on_node failed for GPU worker");
        }
    }
#endif
    T merged = identity<T>(c.op);
    std::ifstream input;
    if (c.storage == StoragePolicy::FileStream) {
        input.open(c.dataset, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open dataset for GPU stream");
        input.seekg(static_cast<std::streamoff>(range.offset * sizeof(T)), std::ios::beg);
    }
    std::size_t chunk_index = 0;
    double storage_ms = 0.0, memcpy_ms = 0.0, h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;
    std::uint64_t storage_bytes = 0, h2d_bytes = 0, d2h_bytes = 0;
    std::size_t chunks = 0;

    for (std::size_t done = 0; done < range.count; done += pipe.capacity, ++chunk_index) {
        auto& slot = pipe.slots[chunk_index % pipe.slots.size()];
        if (slot.pending) {
            CUDA_CHECK(cudaStreamSynchronize(slot.stream));
            merged = combine(merged, *slot.h_output, c.op);
            slot.pending = false;
        }
        const std::size_t n = std::min(pipe.capacity, range.count - done);
        if (c.storage == StoragePolicy::FileStream) {
            storage_ms += timed_ms([&] {
                input.read(reinterpret_cast<char*>(slot.staging.ptr), static_cast<std::streamsize>(n * sizeof(T)));
                if (input.gcount() != static_cast<std::streamsize>(n * sizeof(T))) {
                    throw std::runtime_error("short GPU file-stream read");
                }
            });
            storage_bytes += n * sizeof(T);
        } else {
            memcpy_ms += timed_ms([&] {
                std::memcpy(slot.staging.ptr, host_base + range.offset + done, n * sizeof(T));
            });
        }
        const auto enqueue_start = Clock::now();
        CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.staging.ptr, n * sizeof(T), cudaMemcpyHostToDevice, slot.stream));
        cub_reduce(slot.temp, slot.temp_bytes, slot.d_input, slot.d_output, n, c.op, slot.stream);
        CUDA_CHECK(cudaMemcpyAsync(slot.h_output, slot.d_output, sizeof(T), cudaMemcpyDeviceToHost, slot.stream));
        const auto enqueue_end = Clock::now();
        (void)enqueue_start; (void)enqueue_end;
        h2d_bytes += n * sizeof(T);
        d2h_bytes += sizeof(T);
        slot.pending = true;
        ++chunks;
    }
    for (auto& slot : pipe.slots) {
        if (slot.pending) {
            const auto sync_start = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(slot.stream));
            const double sync_ms = std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count();
            // For an overlapped pipeline, per-stage wall time cannot be uniquely decomposed.
            // Attribute completion wait to kernel bucket and keep bytes exact.
            kernel_ms += sync_ms;
            merged = combine(merged, *slot.h_output, c.op);
            slot.pending = false;
        }
    }

    {
        std::scoped_lock lock(sample_mutex);
        s.storage_read_ms += storage_ms;
        s.host_memcpy_ms += memcpy_ms;
        s.h2d_ms += h2d_ms;
        s.kernel_ms += kernel_ms;
        s.d2h_ms += d2h_ms;
        s.storage_read_bytes += storage_bytes;
        s.h2d_bytes += h2d_bytes;
        s.d2h_bytes += d2h_bytes;
        s.chunks += chunks;
        s.numa_applied = s.numa_applied || pipe.numa_applied;
    }
    (void)gpu_index;
    return merged;
}

template<class T>
std::vector<Sample> run_pinned_direct(const Config& c, T& result) {
    auto host = read_whole<T>(c);
    const std::size_t bytes = c.count * sizeof(T);
    CUDA_CHECK(cudaHostRegister(host.data(), bytes, cudaHostRegisterPortable));
    const std::size_t capacity = std::min(c.count, c.chunk_elements);
    CUDA_CHECK(cudaSetDevice(c.gpu_ids[0]));
    T* d_input = nullptr;
    T* d_output = nullptr;
    T* h_output = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cudaStream_t stream{};
    CUDA_CHECK(cudaMalloc(&d_input, capacity * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(cudaHostAlloc(&h_output, sizeof(T), cudaHostAllocPortable));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cub_query(d_input, d_output, capacity, c.op, stream, temp, temp_bytes);

    auto one = [&]() {
        Sample s;
        const auto start = Clock::now();
        T merged = identity<T>(c.op);
        for (std::size_t offset = 0; offset < c.count; offset += capacity) {
            const std::size_t n = std::min(capacity, c.count - offset);
            s.h2d_ms += timed_ms([&] {
                CUDA_CHECK(cudaMemcpyAsync(d_input, host.data() + offset, n * sizeof(T), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            s.kernel_ms += timed_ms([&] {
                cub_reduce(temp, temp_bytes, d_input, d_output, n, c.op, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            s.d2h_ms += timed_ms([&] {
                CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });
            s.h2d_bytes += n * sizeof(T);
            s.d2h_bytes += sizeof(T);
            merged = combine(merged, *h_output, c.op);
            ++s.chunks;
        }
        result = merged;
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());
    if (temp) cudaFree(temp);
    cudaFree(d_input); cudaFree(d_output); cudaFreeHost(h_output); cudaStreamDestroy(stream);
    CUDA_CHECK(cudaHostUnregister(host.data()));
    return samples;
}

template<class T>
std::vector<Sample> run_topology(const Config& c, T& result) {
    std::vector<T> host;
    if (c.storage == StoragePolicy::HostResident) host = read_whole<T>(c);
    const T* host_base = host.empty() ? nullptr : host.data();

    std::size_t cpu_count = 0;
    if (c.mode == Mode::HybridCpuGpu) {
        cpu_count = static_cast<std::size_t>(static_cast<double>(c.count) * c.cpu_fraction);
        cpu_count = std::min(cpu_count, c.count);
    }
    const Range cpu_range{0, cpu_count};
    const auto gpu_ranges = equal_partition(cpu_count, c.count - cpu_count, c.gpu_ids.size());

    std::vector<std::unique_ptr<GpuPipeline<T>>> pipes;
    pipes.reserve(c.gpu_ids.size());
    const std::size_t capacity = std::min(c.count, c.chunk_elements);
    for (std::size_t i = 0; i < c.gpu_ids.size(); ++i) {
        const int node = i < c.gpu_numa_nodes.size() ? c.gpu_numa_nodes[i] : -1;
        pipes.push_back(std::make_unique<GpuPipeline<T>>(
            c.gpu_ids[i], node, capacity, c.streams, c.op, c.numa_strict));
    }

    auto one = [&]() {
        Sample s;
        std::mutex sample_mutex;
        const auto start = Clock::now();
        std::vector<T> gpu_values(c.gpu_ids.size(), identity<T>(c.op));
        std::vector<std::exception_ptr> errors(c.gpu_ids.size());
        std::vector<std::thread> gpu_threads;
        gpu_threads.reserve(c.gpu_ids.size());

        for (std::size_t i = 0; i < c.gpu_ids.size(); ++i) {
            gpu_threads.emplace_back([&, i] {
                try {
                    gpu_values[i] = reduce_gpu_range(
                        c, gpu_ranges[i], host_base, static_cast<int>(i), *pipes[i], s, sample_mutex);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }

        T cpu_value = identity<T>(c.op);
        if (c.mode == Mode::HybridCpuGpu && cpu_range.count > 0) {
            Sample cpu_sample;
            if (c.storage == StoragePolicy::HostResident) {
                const auto cpu_start = Clock::now();
                cpu_value = cpu_reduce_memory(host_base + cpu_range.offset, cpu_range.count, c.op, c.cpu_threads);
                cpu_sample.cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - cpu_start).count();
                cpu_sample.cpu_elements = cpu_range.count;
            } else {
                cpu_value = cpu_reduce_file<T>(c, cpu_range, c.cpu_threads, c.chunk_elements, cpu_sample);
            }
            std::scoped_lock lock(sample_mutex);
            s.storage_read_ms += cpu_sample.storage_read_ms;
            s.storage_read_bytes += cpu_sample.storage_read_bytes;
            s.cpu_ms += cpu_sample.cpu_ms;
            s.cpu_elements += cpu_sample.cpu_elements;
            s.chunks += cpu_sample.chunks;
        }

        for (auto& t : gpu_threads) t.join();
        for (const auto& e : errors) if (e) std::rethrow_exception(e);
        T merged = cpu_value;
        for (const auto& v : gpu_values) merged = combine(merged, v, c.op);
        result = merged;
        s.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return s;
    };

    for (int i = 0; i < c.warmup; ++i) (void)one();
    std::vector<Sample> samples;
    for (int i = 0; i < c.repetitions; ++i) samples.push_back(one());
    return samples;
}

const char* mode_name(Mode m) {
    if (m == Mode::PinnedDirect) return "pinned_direct";
    if (m == Mode::MultiGpuAsync) return "multi_gpu_async";
    return "hybrid_cpu_gpu";
}

const char* storage_name(StoragePolicy p) {
    return p == StoragePolicy::HostResident ? "host_resident" : "file_stream";
}

template<class T>
void execute(const Config& c) {
    const auto expected = static_cast<std::uintmax_t>(c.count) * sizeof(T);
    if (!std::filesystem::exists(c.dataset) || std::filesystem::file_size(c.dataset) != expected) {
        throw std::runtime_error("dataset file size does not match --count/--dtype");
    }
    T result{};
    std::vector<Sample> samples = c.mode == Mode::PinnedDirect
        ? run_pinned_direct<T>(c, result)
        : run_topology<T>(c, result);

    auto mean_d = [&](auto member) {
        double v = 0.0; for (const auto& s : samples) v += member(s);
        return v / static_cast<double>(samples.size());
    };
    auto mean_u = [&](auto member) {
        std::uint64_t v = 0; for (const auto& s : samples) v += member(s);
        return v / samples.size();
    };
    auto mean_z = [&](auto member) {
        std::size_t v = 0; for (const auto& s : samples) v += member(s);
        return v / samples.size();
    };
    bool numa_applied = false;
    for (const auto& s : samples) numa_applied = numa_applied || s.numa_applied;

    std::cout
        << "{\"event\":\"topology_stream_result\""
        << ",\"mode\":\"" << mode_name(c.mode) << "\""
        << ",\"storage_policy\":\"" << storage_name(c.storage) << "\""
        << ",\"gpu_count\":" << c.gpu_ids.size()
        << ",\"cpu_fraction\":" << c.cpu_fraction
        << ",\"cpu_threads\":" << c.cpu_threads
        << ",\"chunk_elements\":" << c.chunk_elements
        << ",\"streams\":" << c.streams
        << ",\"mean_total_ms\":" << mean_d([](const Sample& s){ return s.total_ms; })
        << ",\"mean_storage_read_ms\":" << mean_d([](const Sample& s){ return s.storage_read_ms; })
        << ",\"mean_host_memcpy_ms\":" << mean_d([](const Sample& s){ return s.host_memcpy_ms; })
        << ",\"mean_h2d_ms\":" << mean_d([](const Sample& s){ return s.h2d_ms; })
        << ",\"mean_kernel_ms\":" << mean_d([](const Sample& s){ return s.kernel_ms; })
        << ",\"mean_d2h_ms\":" << mean_d([](const Sample& s){ return s.d2h_ms; })
        << ",\"mean_cpu_ms\":" << mean_d([](const Sample& s){ return s.cpu_ms; })
        << ",\"mean_storage_read_bytes\":" << mean_u([](const Sample& s){ return s.storage_read_bytes; })
        << ",\"mean_h2d_bytes\":" << mean_u([](const Sample& s){ return s.h2d_bytes; })
        << ",\"mean_d2h_bytes\":" << mean_u([](const Sample& s){ return s.d2h_bytes; })
        << ",\"mean_chunks\":" << mean_z([](const Sample& s){ return s.chunks; })
        << ",\"mean_cpu_elements\":" << mean_z([](const Sample& s){ return s.cpu_elements; })
        << ",\"numa_requested\":" << (!c.gpu_numa_nodes.empty() ? "true" : "false")
        << ",\"numa_applied\":" << (numa_applied ? "true" : "false")
        << ",\"result\":" << result
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto c = parse_cli(argc, argv);
        switch (c.dtype) {
            case DType::Int32: execute<std::int32_t>(c); break;
            case DType::Int64: execute<std::int64_t>(c); break;
            case DType::Float32: execute<float>(c); break;
            case DType::Float64: execute<double>(c); break;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "topology-stream-worker: " << e.what() << "\n";
        return 2;
    }
}
