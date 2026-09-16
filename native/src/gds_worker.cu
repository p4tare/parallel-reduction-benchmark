#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cufile.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using Clock = std::chrono::steady_clock;
#define CUDA_CHECK(expr) do { cudaError_t e=(expr); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define CUFILE_CHECK(expr) do { CUfileError_t e=(expr); if(e.err!=CU_FILE_SUCCESS) throw std::runtime_error(std::string(#expr)+" failed, cuFile error="+std::to_string(e.err)); } while(0)

enum class DType { Int32, Int64, Float32, Float64 };
enum class Op { Sum, Min, Max };
struct Config { std::filesystem::path dataset; DType dtype{DType::Float32}; Op op{Op::Sum}; std::size_t count{0}; std::size_t chunk_elements{16u<<20}; int device{0}; int repetitions{3}; };

DType dtype(std::string_view s){ if(s=="int32")return DType::Int32; if(s=="int64")return DType::Int64; if(s=="float32")return DType::Float32; if(s=="float64")return DType::Float64; throw std::invalid_argument("dtype"); }
Op op(std::string_view s){ if(s=="sum")return Op::Sum; if(s=="min")return Op::Min; if(s=="max")return Op::Max; throw std::invalid_argument("operation"); }
Config parse(int argc,char**argv){ Config c; for(int i=1;i<argc;++i){ std::string a=argv[i]; auto next=[&]{ if(++i>=argc)throw std::invalid_argument("missing "+a); return std::string(argv[i]);}; if(a=="--dataset")c.dataset=next(); else if(a=="--dtype")c.dtype=dtype(next()); else if(a=="--operation")c.op=op(next()); else if(a=="--count")c.count=std::stoull(next()); else if(a=="--chunk-elements")c.chunk_elements=std::stoull(next()); else if(a=="--device")c.device=std::stoi(next()); else if(a=="--repetitions")c.repetitions=std::stoi(next()); else throw std::invalid_argument("unknown "+a);} if(c.dataset.empty()||!c.count||!c.chunk_elements)throw std::invalid_argument("dataset/count/chunk required"); return c; }

template<class T>T identity(Op o){ if(o==Op::Sum)return T{0}; if(o==Op::Min)return std::numeric_limits<T>::max(); return std::numeric_limits<T>::lowest(); }
template<class T>T combine(T a,T b,Op o){ if(o==Op::Sum)return a+b; if(o==Op::Min)return std::min(a,b); return std::max(a,b); }
template<class T>void query(const T*in,T*out,std::size_t n,Op o,cudaStream_t st,void*&tmp,std::size_t&bytes){ bytes=0; if(o==Op::Sum)CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,bytes,in,out,n,st)); else if(o==Op::Min)CUDA_CHECK(cub::DeviceReduce::Min(nullptr,bytes,in,out,n,st)); else CUDA_CHECK(cub::DeviceReduce::Max(nullptr,bytes,in,out,n,st)); if(bytes)CUDA_CHECK(cudaMalloc(&tmp,bytes)); }
template<class T>void reduce(void*tmp,std::size_t bytes,const T*in,T*out,std::size_t n,Op o,cudaStream_t st){ if(o==Op::Sum)CUDA_CHECK(cub::DeviceReduce::Sum(tmp,bytes,in,out,n,st)); else if(o==Op::Min)CUDA_CHECK(cub::DeviceReduce::Min(tmp,bytes,in,out,n,st)); else CUDA_CHECK(cub::DeviceReduce::Max(tmp,bytes,in,out,n,st)); }

template<class T>void execute(const Config& c){
    CUDA_CHECK(cudaSetDevice(c.device));
    const std::size_t cap=std::min(c.count,c.chunk_elements); const std::size_t chunk_bytes=cap*sizeof(T);
    int fd=::open(c.dataset.c_str(),O_RDONLY|O_DIRECT); if(fd<0)throw std::runtime_error("open(O_DIRECT) failed");
    CUfileDescr_t desc{}; desc.handle.fd=fd; desc.type=CU_FILE_HANDLE_TYPE_OPAQUE_FD; CUfileHandle_t handle{};
    CUFILE_CHECK(cuFileDriverOpen()); CUFILE_CHECK(cuFileHandleRegister(&handle,&desc));
    T *din=nullptr,*dout=nullptr; void*temp=nullptr; std::size_t temp_bytes=0; cudaStream_t st{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking)); CUDA_CHECK(cudaMalloc(&din,chunk_bytes)); CUDA_CHECK(cudaMalloc(&dout,sizeof(T))); query(din,dout,cap,c.op,st,temp,temp_bytes);
    for(int rep=0;rep<c.repetitions;++rep){ T merged=identity<T>(c.op); double io_ms=0,kernel_ms=0,d2h_ms=0; std::uint64_t read_bytes=0; std::size_t chunks=0; auto total0=Clock::now();
        for(std::size_t off=0;off<c.count;off+=cap){ std::size_t n=std::min(cap,c.count-off); std::size_t bytes=n*sizeof(T); auto a=Clock::now(); ssize_t got=cuFileRead(handle,din,bytes,static_cast<off_t>(off*sizeof(T)),0); if(got<0||static_cast<std::size_t>(got)!=bytes)throw std::runtime_error("cuFileRead short/error"); CUDA_CHECK(cudaDeviceSynchronize()); io_ms+=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); read_bytes+=bytes;
            a=Clock::now(); reduce(temp,temp_bytes,din,dout,n,c.op,st); CUDA_CHECK(cudaStreamSynchronize(st)); kernel_ms+=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); T partial{}; a=Clock::now(); CUDA_CHECK(cudaMemcpy(&partial,dout,sizeof(T),cudaMemcpyDeviceToHost)); d2h_ms+=std::chrono::duration<double,std::milli>(Clock::now()-a).count(); merged=combine(merged,partial,c.op); ++chunks; }
        double total_ms=std::chrono::duration<double,std::milli>(Clock::now()-total0).count();
        std::cout<<"{\"event\":\"gds_result\",\"status\":\"ok\",\"device\":"<<c.device<<",\"count\":"<<c.count<<",\"element_bytes\":"<<sizeof(T)<<",\"chunk_elements\":"<<cap<<",\"chunks\":"<<chunks<<",\"storage_to_device_bytes\":"<<read_bytes<<",\"storage_to_device_ms\":"<<io_ms<<",\"kernel_ms\":"<<kernel_ms<<",\"d2h_ms\":"<<d2h_ms<<",\"total_ms\":"<<total_ms<<",\"result\":"<<merged<<"}\n";
    }
    if(temp)cudaFree(temp); cudaFree(din); cudaFree(dout); cudaStreamDestroy(st); cuFileHandleDeregister(handle); cuFileDriverClose(); ::close(fd);
}
}

int main(int argc,char**argv){ try{ auto c=parse(argc,argv); switch(c.dtype){case DType::Int32:execute<std::int32_t>(c);break;case DType::Int64:execute<std::int64_t>(c);break;case DType::Float32:execute<float>(c);break;case DType::Float64:execute<double>(c);break;} return 0;}catch(const std::exception&e){std::cerr<<"gds-worker: "<<e.what()<<"\n";return 2;} }
