#include <iostream>
#include <chrono>
#include <thread>
#include <cstdint>
#include <omp.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <stdio.h>
#include <algorithm>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "{\"error\": \"CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << "\"}" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__device__ inline void customAtomicAdd(int32_t* address, int32_t val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(float* address, float val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(double* address, double val) { atomicAdd(address, val); }
__device__ inline void customAtomicAdd(int64_t* address, int64_t val) {
    atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

// Inject user algorithm
#include "kernel.cuh"

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

    if (data_path.empty()) { std::cerr << "{\"error\": \"No data path\"}" << std::endl; return 1; }

    struct stat sb;
    if (stat(data_path.c_str(), &sb) == -1) { std::cerr << "{\"error\": \"Failed to stat file\"}" << std::endl; return 1; }
    size_t total_elements = sb.st_size / sizeof(DATA_TYPE);

    // Dynamic Chunk Sizing - Ask GPU for free memory
    size_t free_vram, total_vram;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
    
    // We use 50% of free VRAM to be safe, but cap at 1GB chunks to save host RAM
    size_t max_chunk_bytes = free_vram / 2;
    size_t max_elements_per_chunk = max_chunk_bytes / sizeof(DATA_TYPE);
    size_t hard_limit_elements = 250000000; // ~1GB max chunk
    
    if (max_elements_per_chunk > hard_limit_elements) {
        max_elements_per_chunk = hard_limit_elements;
    }

    size_t current_chunk_size = std::min(total_elements, max_elements_per_chunk);
    
    // Allocate CPU RAM buffer
    DATA_TYPE* h_buffer = new DATA_TYPE[current_chunk_size];
    FILE* file = fopen(data_path.c_str(), "rb");
    if (!file) { std::cerr << "{\"error\": \"Failed to open file\"}" << std::endl; delete[] h_buffer; return 1; }

    double total_time_us = 0;
    double final_result = 0;
    size_t elements_read = 0;

    // Out-of-Core Processing Loop
    while (elements_read < total_elements) {
        size_t elements_to_read = std::min(current_chunk_size, total_elements - elements_read);
        
        // Disk I/O is NOT timed
        fread(h_buffer, sizeof(DATA_TYPE), elements_to_read, file);
        
        double chunk_time = 0;
        double chunk_result = 0;
        
        // Timer runs inside this function, covering RAM->VRAM, kernel, and VRAM->RAM
        execute_algorithm(h_buffer, elements_to_read, reps, warmup, block_size, dedicated_threads, chunk_time, chunk_result);
        
        total_time_us += chunk_time;
        final_result += chunk_result;
        elements_read += elements_to_read;
    }

    fclose(file);
    delete[] h_buffer;

    std::cout << "{";
    std::cout << "\"cpp_cpu_time_us\": " << total_time_us << ", ";
    std::cout << "\"gpu_kernel_time_us\": " << total_time_us << ", ";
    std::cout << "\"reduction_result\": " << final_result << ", ";
    std::cout << "\"openmp_max_threads\": " << omp_get_max_threads();
    std::cout << "}" << std::endl;

    return 0;
}