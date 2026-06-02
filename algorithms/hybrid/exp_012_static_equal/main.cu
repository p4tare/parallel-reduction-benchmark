#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cstdint>
#include <omp.h>
#include <cuda_runtime.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Custom atomic wrappers
__device__ inline void customAtomicAdd(int32_t* address, int32_t val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(float* address, float val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(double* address, double val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(int64_t* address, int64_t val) {
    atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

// Simple GPU reduction kernel
template <typename T>
__global__ void gpu_reduce_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    T local_sum = 0;
    for (size_t i = idx; i < n; i += stride) {
        local_sum += input[i];
    }
    
    // Fallback to atomic for block aggregation to keep it simple macro-wise
    if (local_sum != 0) {
        customAtomicAdd(output, local_sum);
    }
}

int main(int argc, char** argv) {
    std::string data_path = "";
    int reps = 10;
    int warmup = 3;
    bool dedicated_threads = false;
    int block_size = 256;

    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--data" && i+1 < argc) data_path = argv[++i];
        else if(arg == "--reps" && i+1 < argc) reps = std::stoi(argv[++i]);
        else if(arg == "--warmup" && i+1 < argc) warmup = std::stoi(argv[++i]);
        else if(arg == "--dedicated-threads" && i+1 < argc) dedicated_threads = (std::stoi(argv[++i]) == 1);
        else if(arg == "--block-size" && i+1 < argc) block_size = std::stoi(argv[++i]);
    }

    if (data_path.empty()) return 1;

    std::ifstream file(data_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return 1;
    
    size_t file_size = file.tellg();
    size_t n = file_size / sizeof(DATA_TYPE);
    file.seekg(0, std::ios::beg);
    
    std::vector<DATA_TYPE> h_input(n);
    file.read(reinterpret_cast<char*>(h_input.data()), file_size);
    file.close();

    // EXP_012 MACRO STATIC EQUAL SPLIT: 50% GPU, 50% CPU
    size_t n_gpu = n / 2;
    size_t n_cpu = n - n_gpu;
    size_t cpu_offset = n_gpu; // CPU starts where GPU ends

    DATA_TYPE* d_input = nullptr;
    DATA_TYPE* d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, n_gpu * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n_gpu * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));

    int num_blocks = (n_gpu + block_size - 1) / block_size;
    if (num_blocks > 1024) num_blocks = 1024; // Limit blocks for grid-stride

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DATA_TYPE final_result = 0;
    double total_hybrid_time_ms = 0;

    // Measurement Loop
    for (int r = 0; r < warmup + reps; r++) {
        DATA_TYPE h_gpu_output = 0;
        DATA_TYPE h_cpu_output = 0;
        
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(DATA_TYPE), stream));

        auto start_time = std::chrono::high_resolution_clock::now();

        if (dedicated_threads) {
            // Paradigm 1: Dedicated CPU thread manages GPU, main thread does CPU OpenMP work
            std::thread gpu_thread([&]() {
                gpu_reduce_kernel<<<num_blocks, block_size, 0, stream>>>(d_input, d_output, n_gpu);
                CUDA_CHECK(cudaMemcpyAsync(&h_gpu_output, d_output, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            });

            // CPU Worker Pool (Main thread + others)
            #pragma omp parallel for reduction(+:h_cpu_output)
            for (size_t i = cpu_offset; i < n; i++) {
                h_cpu_output += h_input[i];
            }

            gpu_thread.join(); // Wait for GPU thread to finish
        } 
        else {
            // Paradigm 2: Asynchronous Launch (No dedicated thread blocked)
            gpu_reduce_kernel<<<num_blocks, block_size, 0, stream>>>(d_input, d_output, n_gpu);
            CUDA_CHECK(cudaMemcpyAsync(&h_gpu_output, d_output, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, stream));

            // CPU Worker Pool computes while GPU works in background
            #pragma omp parallel for reduction(+:h_cpu_output)
            for (size_t i = cpu_offset; i < n; i++) {
                h_cpu_output += h_input[i];
            }

            // Sync at the end
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (r >= warmup) {
            std::chrono::duration<double> duration = end_time - start_time;
            total_hybrid_time_ms += (duration.count() * 1000.0);
        }
        
        // Final addition
        if (r == warmup + reps - 1) {
            final_result = h_gpu_output + h_cpu_output;
        }
    }

    double avg_hybrid_time_us = (total_hybrid_time_ms * 1000.0) / reps;

    std::cout << "{";
    std::cout << "\"cpp_cpu_time_us\": " << avg_hybrid_time_us << ", ";
    std::cout << "\"gpu_kernel_time_us\": " << avg_hybrid_time_us << ", "; // Same, since it's hybrid wall-time
    std::cout << "\"reduction_result\": " << static_cast<double>(final_result) << ", ";
    std::cout << "\"openmp_max_threads\": " << omp_get_max_threads();
    std::cout << "}" << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}