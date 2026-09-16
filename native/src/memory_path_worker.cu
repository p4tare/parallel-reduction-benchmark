#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
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
    if (_e != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(_e)); \
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

std::string mode_name(Mode m) {
    switch (m) {
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
    throw std::invalid_argument("unsupported dtype");
}

Op parse_op(std::string_view s) {
    if (s == "sum") return Op::Sum;
    if (s == "min") return Op::Min;
    if (s == "max") return Op::Max;
    throw std::invalid_argument("unsupported operation");
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
    throw std::invalid_argument("unsupported mode");
}

Config parse_cli(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[i];
        };
        if (a == "--dataset") c.dataset = need();
        else if (a == "--dtype") c.dtype = parse_dtype(need());
        else if (a == "--operation") c.op = parse_op(need());
        else if (a == "--mode") c.mode = parse_mode(need());
        else if (a == "--count") c.count = std::stoull(need());
        else if (a == "--chunk-elements") c.chunk_elements = std::stoull(need());
        else if (a == "--streams") c.streams = std::stoi(need());
        else if (a == "--device") c.device = std::stoi(need());
        else if (a == "--reuse-count") c.reuse_count = std::stoi(need());
        else if (a == "--warmup") c.warmup = std::stoi(need());
        else if (a == "--repetitions") c.repetitions = std::stoi(need());
        else if (a == "--cuda-graphs") c.use_cuda_graphs = true;
        else throw std::invalid_argument("unknown argument: " + a);
    }
    if (c.dataset.empty() || c.count == 0) throw std::invalid_argument("--dataset and --count are required");
    if (c.chunk_elements == 0 || c.streams < 1 || c.reuse_count < 1 || c.warmup < 0 || c.repetitions < 1)
        throw std::invalid_argument("invalid numeric option");
    return c;
}

template<class T> T identity(Op op) {
    if (op == Op::Sum) return T{0};
    if (op == Op::Min) return std::numeric_limits<T>::max();
    return std::numeric_limits<T>::lowest();
}

template<class T> T combine(T a, T b, Op op) {
    if (op == Op::Sum) return a + b;
    if (op == Op::Min) return std::min(a, b);
    return std::max(a, b);
}

template<class T>
void cub_query_and_alloc(const T* in, T* out, std::size_t n, Op op, cudaStream_t stream,
                         void*& temp, std::size_t& temp_bytes) {
    temp_bytes = 0;
    if (op == Op::Sum) CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, in, out, n, stream));
    else if (op == Op::Min) CUDA_CHECK(cub::DeviceReduce::Min(nullptr, temp_bytes, in, out, n, stream));
    else CUDA_CHECK(cub::DeviceReduce::Max(nullptr, temp_bytes, in, out, n, stream));
    if (temp_bytes) CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
}

template<class T>
void cub_reduce(void* temp, std::size_t temp_bytes, const T* in, T* out, std::size_t n, Op op, cudaStream_t stream) {
    if (op == Op::Sum) CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, in, out, n, stream));
    else if (op == Op::Min) CUDA_CHECK(cub::DeviceReduce::Min(temp, temp_bytes, in, out, n, stream));
    else CUDA_CHECK(cub::DeviceReduce::Max(temp, temp_bytes, in, out, n, stream));
}

struct Sample {
    double total_ms{0};
    double h2d_ms{0};
    double kernel_ms{0};
    double d2h_ms{0};
    std::uint64_t host_to_device_bytes{0};
    std::uint64_t device_to_host_bytes{0};
    std::uint64_t remote_host_read_bytes{0};
    std::size_t chunks{0};
};

template<class T>
std::vector<T> read_pageable(const Config& c) {
    std::vector<T> h(c.count);
    std::ifstream f(c.dataset, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open dataset");
    f.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(c.count * sizeof(T)));
    if (!f) throw std::runtime_error("cannot read complete dataset");
    return h;
}

template<class T>
void read_exact(const Config& c, T* p) {
    std::ifstream f(c.dataset, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open dataset");
    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(c.count * sizeof(T)));
    if (!f) throw std::runtime_error("cannot read complete dataset");
}

template<class T>
Sample explicit_sync(const Config& c, const T* h, T& result, bool include_upload) {
    Sample s;
    T *di=nullptr, *do_=nullptr;
    void* temp=nullptr; std::size_t temp_bytes=0;
    cudaStream_t st{}; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&di, c.count*sizeof(T))); CUDA_CHECK(cudaMalloc(&do_, sizeof(T)));
    cub_query_and_alloc(di, do_, c.count, c.op, st, temp, temp_bytes);
    auto t0=Clock::now();
    if (include_upload) {
        auto a=Clock::now(); CUDA_CHECK(cudaMemcpyAsync(di,h,c.count*sizeof(T),cudaMemcpyHostToDevice,st)); CUDA_CHECK(cudaStreamSynchronize(st));
        s.h2d_ms=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.host_to_device_bytes=c.count*sizeof(T);
    }
    for(int k=0;k<c.reuse_count;++k) {
        auto a=Clock::now(); cub_reduce(temp,temp_bytes,di,do_,c.count,c.op,st); CUDA_CHECK(cudaStreamSynchronize(st));
        s.kernel_ms += std::chrono::duration<double,std::milli>(Clock::now()-a).count();
    }
    auto a=Clock::now(); CUDA_CHECK(cudaMemcpyAsync(&result,do_,sizeof(T),cudaMemcpyDeviceToHost,st)); CUDA_CHECK(cudaStreamSynchronize(st));
    s.d2h_ms=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.device_to_host_bytes=sizeof(T);
    s.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count(); s.chunks=1;
    if(temp) cudaFree(temp); cudaFree(di); cudaFree(do_); cudaStreamDestroy(st); return s;
}

template<class T>
Sample chunked_sync(const Config& c, const T* h, T& result) {
    Sample s; const std::size_t cap=std::min(c.count,c.chunk_elements);
    T *di=nullptr,*do_=nullptr; void* temp=nullptr; std::size_t temp_bytes=0;
    cudaStream_t st{}; CUDA_CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&di,cap*sizeof(T))); CUDA_CHECK(cudaMalloc(&do_,sizeof(T)));
    cub_query_and_alloc(di,do_,cap,c.op,st,temp,temp_bytes);
    auto total0=Clock::now();
    for(int reuse=0; reuse<c.reuse_count; ++reuse) {
        T merged=identity<T>(c.op);
        for(std::size_t off=0; off<c.count; off+=cap) {
            const std::size_t n=std::min(cap,c.count-off); T partial{};
            auto a=Clock::now(); CUDA_CHECK(cudaMemcpyAsync(di,h+off,n*sizeof(T),cudaMemcpyHostToDevice,st)); CUDA_CHECK(cudaStreamSynchronize(st));
            s.h2d_ms += std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.host_to_device_bytes += n*sizeof(T);
            a=Clock::now(); cub_reduce(temp,temp_bytes,di,do_,n,c.op,st); CUDA_CHECK(cudaStreamSynchronize(st));
            s.kernel_ms += std::chrono::duration<double,std::milli>(Clock::now()-a).count();
            a=Clock::now(); CUDA_CHECK(cudaMemcpyAsync(&partial,do_,sizeof(T),cudaMemcpyDeviceToHost,st)); CUDA_CHECK(cudaStreamSynchronize(st));
            s.d2h_ms += std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.device_to_host_bytes += sizeof(T);
            merged=combine(merged,partial,c.op); ++s.chunks;
        }
        result=merged;
    }
    s.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total0).count();
    if(temp) cudaFree(temp); cudaFree(di); cudaFree(do_); cudaStreamDestroy(st); return s;
}

template<class T>
Sample async_pipeline(const Config& c, const T* pageable, T& result) {
    Sample s; const std::size_t cap=std::min(c.count,c.chunk_elements); const int ns=c.streams;
    struct Slot { cudaStream_t st{}; T* h=nullptr; T* di=nullptr; T* dout=nullptr; void* temp=nullptr; std::size_t temp_bytes=0; T partial{}; };
    std::vector<Slot> slots(ns);
    for(auto& x:slots){ CUDA_CHECK(cudaStreamCreateWithFlags(&x.st,cudaStreamNonBlocking)); CUDA_CHECK(cudaHostAlloc(&x.h,cap*sizeof(T),cudaHostAllocPortable)); CUDA_CHECK(cudaMalloc(&x.di,cap*sizeof(T))); CUDA_CHECK(cudaMalloc(&x.dout,sizeof(T))); cub_query_and_alloc(x.di,x.dout,cap,c.op,x.st,x.temp,x.temp_bytes); }
    auto total0=Clock::now();
    for(int reuse=0; reuse<c.reuse_count; ++reuse) {
        T merged=identity<T>(c.op); std::size_t chunk=0;
        for(std::size_t off=0; off<c.count; off+=cap,++chunk) {
            Slot& x=slots[chunk%ns]; CUDA_CHECK(cudaStreamSynchronize(x.st));
            if(chunk>=static_cast<std::size_t>(ns)) merged=combine(merged,x.partial,c.op);
            const std::size_t n=std::min(cap,c.count-off);
            std::memcpy(x.h,pageable+off,n*sizeof(T));
            CUDA_CHECK(cudaMemcpyAsync(x.di,x.h,n*sizeof(T),cudaMemcpyHostToDevice,x.st));
            cub_reduce(x.temp,x.temp_bytes,x.di,x.dout,n,c.op,x.st);
            CUDA_CHECK(cudaMemcpyAsync(&x.partial,x.dout,sizeof(T),cudaMemcpyDeviceToHost,x.st));
            s.host_to_device_bytes += n*sizeof(T); s.device_to_host_bytes += sizeof(T); ++s.chunks;
        }
        const std::size_t total_chunks=(c.count+cap-1)/cap;
        for(std::size_t i=0;i<std::min<std::size_t>(total_chunks,ns);++i){ CUDA_CHECK(cudaStreamSynchronize(slots[i].st)); merged=combine(merged,slots[i].partial,c.op); }
        result=merged;
    }
    s.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total0).count();
    // With overlap, component sums are intentionally not reconstructed from wall clock.
    for(auto& x:slots){ if(x.temp) cudaFree(x.temp); cudaFree(x.di); cudaFree(x.dout); cudaFreeHost(x.h); cudaStreamDestroy(x.st); } return s;
}

template<class T>
Sample zero_copy(const Config& c, T& result) {
    Sample s; T* h=nullptr; T* mapped=nullptr; T* dout=nullptr; void* temp=nullptr; std::size_t temp_bytes=0; cudaStream_t st{};
    CUDA_CHECK(cudaHostAlloc(&h,c.count*sizeof(T),cudaHostAllocMapped|cudaHostAllocPortable)); read_exact(c,h);
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped,h,0)); CUDA_CHECK(cudaMalloc(&dout,sizeof(T))); CUDA_CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking)); cub_query_and_alloc(mapped,dout,c.count,c.op,st,temp,temp_bytes);
    auto t0=Clock::now();
    for(int k=0;k<c.reuse_count;++k){ auto a=Clock::now(); cub_reduce(temp,temp_bytes,mapped,dout,c.count,c.op,st); CUDA_CHECK(cudaStreamSynchronize(st)); s.kernel_ms+=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.remote_host_read_bytes += c.count*sizeof(T); }
    auto a=Clock::now(); CUDA_CHECK(cudaMemcpyAsync(&result,dout,sizeof(T),cudaMemcpyDeviceToHost,st)); CUDA_CHECK(cudaStreamSynchronize(st)); s.d2h_ms=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.device_to_host_bytes=sizeof(T); s.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count(); s.chunks=1;
    if(temp) cudaFree(temp); cudaFree(dout); cudaFreeHost(h); cudaStreamDestroy(st); return s;
}

template<class T>
Sample managed_mode(const Config& c, T& result) {
    Sample s; T* data=nullptr; T* dout=nullptr; void* temp=nullptr; std::size_t temp_bytes=0; cudaStream_t st{};
    CUDA_CHECK(cudaMallocManaged(&data,c.count*sizeof(T))); read_exact(c,data); CUDA_CHECK(cudaMalloc(&dout,sizeof(T))); CUDA_CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    if(c.mode==Mode::ManagedAdvised){ CUDA_CHECK(cudaMemAdvise(data,c.count*sizeof(T),cudaMemAdviseSetReadMostly,c.device)); CUDA_CHECK(cudaMemAdvise(data,c.count*sizeof(T),cudaMemAdviseSetAccessedBy,c.device)); }
    cub_query_and_alloc(data,dout,c.count,c.op,st,temp,temp_bytes); auto t0=Clock::now();
    if(c.mode==Mode::ManagedPrefetch){ auto a=Clock::now(); CUDA_CHECK(cudaMemPrefetchAsync(data,c.count*sizeof(T),c.device,st)); CUDA_CHECK(cudaStreamSynchronize(st)); s.h2d_ms=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.host_to_device_bytes=c.count*sizeof(T); }
    for(int k=0;k<c.reuse_count;++k){ auto a=Clock::now(); cub_reduce(temp,temp_bytes,data,dout,c.count,c.op,st); CUDA_CHECK(cudaStreamSynchronize(st)); s.kernel_ms+=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); }
    auto a=Clock::now(); CUDA_CHECK(cudaMemcpyAsync(&result,dout,sizeof(T),cudaMemcpyDeviceToHost,st)); CUDA_CHECK(cudaStreamSynchronize(st)); s.d2h_ms=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); s.device_to_host_bytes=sizeof(T); s.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count(); s.chunks=1;
    if(temp) cudaFree(temp); cudaFree(dout); cudaFree(data); cudaStreamDestroy(st); return s;
}

template<class T>
Sample hmm_system(const Config& c, T& result) {
    int pageable=0, host_tables=0;
    CUDA_CHECK(cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,c.device));
    CUDA_CHECK(cudaDeviceGetAttribute(&host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,c.device));
    if(!pageable) throw std::runtime_error("HMM/system pageable GPU access unavailable: cudaDevAttrPageableMemoryAccess=0");
    std::unique_ptr<T[]> h(new T[c.count]); read_exact(c,h.get()); T* dout=nullptr; void* temp=nullptr; std::size_t temp_bytes=0; cudaStream_t st{};
    CUDA_CHECK(cudaMalloc(&dout,sizeof(T))); CUDA_CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking)); cub_query_and_alloc(h.get(),dout,c.count,c.op,st,temp,temp_bytes); Sample s; auto t0=Clock::now();
    for(int k=0;k<c.reuse_count;++k){ cub_reduce(temp,temp_bytes,h.get(),dout,c.count,c.op,st); CUDA_CHECK(cudaStreamSynchronize(st)); s.remote_host_read_bytes += c.count*sizeof(T); }
    CUDA_CHECK(cudaMemcpyAsync(&result,dout,sizeof(T),cudaMemcpyDeviceToHost,st)); CUDA_CHECK(cudaStreamSynchronize(st)); s.device_to_host_bytes=sizeof(T); s.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-t0).count(); s.chunks=1;
    if(temp) cudaFree(temp); cudaFree(dout); cudaStreamDestroy(st); (void)host_tables; return s;
}

template<class T>
Sample run_once(const Config& c, T& result) {
    CUDA_CHECK(cudaSetDevice(c.device));
    if(c.mode==Mode::ZeroCopy) return zero_copy<T>(c,result);
    if(c.mode==Mode::ManagedFault || c.mode==Mode::ManagedPrefetch || c.mode==Mode::ManagedAdvised) return managed_mode<T>(c,result);
    if(c.mode==Mode::HmmSystem) return hmm_system<T>(c,result);
    auto h=read_pageable<T>(c);
    if(c.mode==Mode::ExplicitSync) return explicit_sync<T>(c,h.data(),result,true);
    if(c.mode==Mode::DeviceResident) return explicit_sync<T>(c,h.data(),result,true);
    if(c.mode==Mode::ChunkedSync) return chunked_sync<T>(c,h.data(),result);
    if(c.mode==Mode::AsyncPipeline) return async_pipeline<T>(c,h.data(),result);
    throw std::logic_error("unreachable mode");
}

struct Features { int managed=0, concurrent_managed=0, pageable=0, host_tables=0, map_host=0; };
Features features(int d){ Features f; cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,d)); f.map_host=p.canMapHostMemory; CUDA_CHECK(cudaDeviceGetAttribute(&f.managed,cudaDevAttrManagedMemory,d)); CUDA_CHECK(cudaDeviceGetAttribute(&f.concurrent_managed,cudaDevAttrConcurrentManagedAccess,d)); CUDA_CHECK(cudaDeviceGetAttribute(&f.pageable,cudaDevAttrPageableMemoryAccess,d)); CUDA_CHECK(cudaDeviceGetAttribute(&f.host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,d)); return f; }

template<class T>
void execute(const Config& c) {
    for(int i=0;i<c.warmup;++i){ T v{}; (void)run_once<T>(c,v); }
    std::vector<Sample> samples; samples.reserve(c.repetitions); T result{};
    for(int i=0;i<c.repetitions;++i) samples.push_back(run_once<T>(c,result));
    auto avg=[&](auto fn){ double x=0; for(auto&s:samples)x+=fn(s); return x/samples.size(); };
    auto sumu=[&](auto fn){ std::uint64_t x=0; for(auto&s:samples)x+=fn(s); return x/samples.size(); };
    auto feat=features(c.device); cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,c.device));
    std::cout << "{\"event\":\"memory_path_result\",\"mode\":\""<<mode_name(c.mode)<<"\",\"device\":"<<c.device
              <<",\"gpu_name\":\""<<p.name<<"\",\"count\":"<<c.count<<",\"element_bytes\":"<<sizeof(T)
              <<",\"reuse_count\":"<<c.reuse_count<<",\"chunk_elements\":"<<c.chunk_elements<<",\"streams\":"<<c.streams
              <<",\"mean_total_ms\":"<<avg([](auto&s){return s.total_ms;})<<",\"mean_h2d_ms\":"<<avg([](auto&s){return s.h2d_ms;})
              <<",\"mean_kernel_ms\":"<<avg([](auto&s){return s.kernel_ms;})<<",\"mean_d2h_ms\":"<<avg([](auto&s){return s.d2h_ms;})
              <<",\"mean_h2d_bytes\":"<<sumu([](auto&s){return s.host_to_device_bytes;})<<",\"mean_d2h_bytes\":"<<sumu([](auto&s){return s.device_to_host_bytes;})
              <<",\"mean_remote_host_read_bytes\":"<<sumu([](auto&s){return s.remote_host_read_bytes;})
              <<",\"cuda_managed_memory\":"<<feat.managed<<",\"cuda_concurrent_managed_access\":"<<feat.concurrent_managed
              <<",\"cuda_pageable_memory_access\":"<<feat.pageable<<",\"cuda_pageable_uses_host_page_tables\":"<<feat.host_tables
              <<",\"cuda_can_map_host_memory\":"<<feat.map_host<<",\"cuda_graphs_requested\":"<<(c.use_cuda_graphs?"true":"false")
              <<",\"result\":"<<result<<"}\n";
}

} // namespace

int main(int argc,char** argv){
    try{
        Config c=parse_cli(argc,argv);
        switch(c.dtype){
            case DType::Int32: execute<std::int32_t>(c); break;
            case DType::Int64: execute<std::int64_t>(c); break;
            case DType::Float32: execute<float>(c); break;
            case DType::Float64: execute<double>(c); break;
        }
        return 0;
    }catch(const std::exception& e){ std::cerr<<"memory-path-worker: "<<e.what()<<"\n"; return 2; }
}
