#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cuda_runtime.h>

// Fallback macro in case Python fails to inject the type
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

// Helper macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// -------------------------------------------------------------------------
// CUSTOM ATOMIC ADD WRAPPERS (Handles generic types gracefully)
// -------------------------------------------------------------------------
__device__ inline void customAtomicAdd(int32_t* address, int32_t val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(float* address, float val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(double* address, double val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(int64_t* address, int64_t val) {
    atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

// -------------------------------------------------------------------------
// CUDA KERNEL (EXP_001: Global Memory Atomics)
// Purposefully creates massive memory contention as per experiment design.
// -------------------------------------------------------------------------
template <typename T>
__global__ void reduce_global_atomics_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop directly hammering global memory
    for (size_t i = idx; i < n; i += stride) {
        customAtomicAdd(output, input[i]);
    }
}

// -------------------------------------------------------------------------
// GPU EXECUTION WORKER
// -------------------------------------------------------------------------
void run_gpu_work(const std::vector<DATA_TYPE>& h_input, size_t n, int reps, int warmup, int block_size) {
    DATA_TYPE* d_input = nullptr;
    DATA_TYPE* d_output = nullptr;
    
    // Allocate VRAM
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(DATA_TYPE)));
    
    // Copy data to VRAM
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    
    // Grid setup (capped to a reasonable limit for grid-stride loop)
    int num_blocks = (n + block_size - 1) / block_size;
    if (num_blocks > 65535) num_blocks = 65535;
    
    DATA_TYPE h_output = 0;
    
    // 1. WARMUP PHASE
    for(int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(DATA_TYPE)));
        reduce_global_atomics_kernel<<<num_blocks, block_size>>>(d_input, d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 2. MEASUREMENT PHASE
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    
    for(int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(DATA_TYPE)));
        reduce_global_atomics_kernel<<<num_blocks, block_size>>>(d_input, d_output, n);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for all reps to finish
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    // Fetch result
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    
    // 3. CALCULATION
    float gpu_time_total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_total_ms, start, stop));
    
    double avg_gpu_time_us = (gpu_time_total_ms * 1000.0) / reps;
    
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    double avg_cpu_time_us = (cpu_duration.count() * 1000000.0) / reps;
    
    // 4. OUTPUT JSON FOR PYTHON ORCHESTRATOR
    std::cout << "{";
    std::cout << "\"cpp_cpu_time_us\": " << avg_cpu_time_us << ", ";
    std::cout << "\"gpu_kernel_time_us\": " << avg_gpu_time_us << ", ";
    std::cout << "\"reduction_result\": " << static_cast<double>(h_output);
    std::cout << "}" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// -------------------------------------------------------------------------
// MAIN ENTRY POINT
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string data_path = "";
    int reps = 10;
    int warmup = 3;
    bool dedicated_threads = false;
    int block_size = 256;

    // Parse CLI arguments passed by Python Orchestrator
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--data" && i+1 < argc) data_path = argv[++i];
        else if(arg == "--reps" && i+1 < argc) reps = std::stoi(argv[++i]);
        else if(arg == "--warmup" && i+1 < argc) warmup = std::stoi(argv[++i]);
        else if(arg == "--dedicated-threads" && i+1 < argc) dedicated_threads = (std::stoi(argv[++i]) == 1);
        else if(arg == "--block-size" && i+1 < argc) block_size = std::stoi(argv[++i]);
    }

    if (data_path.empty()) {
        std::cerr << "{\"error\": \"No data path provided\"}" << std::endl;
        return 1;
    }

    // Load binary data generated by DataFactory
    std::ifstream file(data_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "{\"error\": \"Failed to open data file\"}" << std::endl;
        return 1;
    }
    
    size_t file_size = file.tellg();
    size_t n = file_size / sizeof(DATA_TYPE);
    file.seekg(0, std::ios::beg);
    
    std::vector<DATA_TYPE> h_input(n);
    file.read(reinterpret_cast<char*>(h_input.data()), file_size);
    file.close();

    // Execute based on hybrid thread strategy
    if (dedicated_threads) {
        // Spawns a dedicated CPU thread to oversee GPU execution while main thread is free
        std::thread gpu_thread(run_gpu_work, std::ref(h_input), n, reps, warmup, block_size);
        gpu_thread.join();
    } else {
        // Inline execution
        run_gpu_work(h_input, n, reps, warmup, block_size);
    }

    return 0;
}