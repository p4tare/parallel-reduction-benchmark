#include <iostream>
#include <chrono>
#include <thread>
#include <cstdint>
#include <omp.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <sstream>
#include <string>

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

// INJECT USER ALGORITHM
// (The kernel.cuh file might optionally define IS_HETEROGENEOUS_AWARE)
#include "kernel.cuh"

// Helper function to parse array of GPUs from string
std::vector<int> parse_gpu_list(std::string gpu_str) {
    std::vector<int> gpus;
    gpu_str.erase(std::remove(gpu_str.begin(), gpu_str.end(), '['), gpu_str.end());
    gpu_str.erase(std::remove(gpu_str.begin(), gpu_str.end(), ']'), gpu_str.end());
    gpu_str.erase(std::remove(gpu_str.begin(), gpu_str.end(), ' '), gpu_str.end());
    
    std::stringstream ss(gpu_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            gpus.push_back(std::stoi(token));
        }
    }
    if (gpus.empty()) gpus.push_back(0); // Fallback
    return gpus;
}

int main(int argc, char** argv) {
    std::string data_path = "";
    std::string gpus_raw = "0";
    int reps = 10;
    int warmup = 3;
    bool dedicated_threads = false;
    int block_size = 256;
    bool do_trace = false; // NEW TRACE FLAG

    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--data" && i+1 < argc) data_path = argv[++i];
        else if(arg == "--gpus" && i+1 < argc) gpus_raw = argv[++i];
        else if(arg == "--reps" && i+1 < argc) reps = std::stoi(argv[++i]);
        else if(arg == "--warmup" && i+1 < argc) warmup = std::stoi(argv[++i]);
        else if(arg == "--dedicated-threads" && i+1 < argc) dedicated_threads = (std::stoi(argv[++i]) == 1);
        else if(arg == "--block-size" && i+1 < argc) block_size = std::stoi(argv[++i]);
        else if(arg == "--trace" && i+1 < argc) do_trace = (std::stoi(argv[++i]) == 1);
    }

    if (data_path.empty()) { std::cerr << "{\"error\": \"No data path\"}" << std::endl; return 1; }

    std::vector<int> target_gpus = parse_gpu_list(gpus_raw);
    int num_gpus = target_gpus.size();

    struct stat sb;
    if (stat(data_path.c_str(), &sb) == -1) { std::cerr << "{\"error\": \"Failed to stat file\"}" << std::endl; return 1; }
    size_t total_elements = sb.st_size / sizeof(DATA_TYPE);

    size_t current_chunk_size = std::min(total_elements, (size_t)250000000);
    
    DATA_TYPE* h_buffer = new DATA_TYPE[current_chunk_size];
    FILE* file = fopen(data_path.c_str(), "rb");
    if (!file) { std::cerr << "{\"error\": \"Failed to open file\"}" << std::endl; delete[] h_buffer; return 1; }

    double total_time_us = 0;
    double final_result = 0;
    size_t elements_read = 0;

    while (elements_read < total_elements) {
        size_t elements_to_read = std::min(current_chunk_size, total_elements - elements_read);
        fread(h_buffer, sizeof(DATA_TYPE), elements_to_read, file);
        
        double max_chunk_time_us = 0;
        double sum_chunk_result = 0;

#ifdef IS_HETEROGENEOUS_AWARE
        int total_workers = 1 + num_gpus; 
        omp_set_num_threads(total_workers);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            bool is_cpu_worker = (thread_id == 0);
            int my_gpu_id = -1;
            
            if (!is_cpu_worker) {
                my_gpu_id = target_gpus[thread_id - 1]; 
                CUDA_CHECK(cudaSetDevice(my_gpu_id));
            }
            
            double thread_time = 0;
            double thread_result = 0;
            
            // ADDED do_trace FLAG TO SIGNATURE
            execute_algorithm(h_buffer, elements_to_read, reps, warmup, block_size, dedicated_threads, 
                              thread_id, total_workers, is_cpu_worker, my_gpu_id, do_trace, thread_time, thread_result);
            
            #pragma omp critical
            {
                sum_chunk_result += thread_result;
                if (thread_time > max_chunk_time_us) {
                    max_chunk_time_us = thread_time; 
                }
            }
        }
#else
        CUDA_CHECK(cudaSetDevice(target_gpus[0])); 
        execute_algorithm(h_buffer, elements_to_read, reps, warmup, block_size, dedicated_threads, 
                          max_chunk_time_us, sum_chunk_result);
#endif

        total_time_us += max_chunk_time_us;
        final_result += sum_chunk_result;
        elements_read += elements_to_read;
    }

    fclose(file);
    delete[] h_buffer;

    std::cout << "{";
    std::cout << "\"cpp_cpu_time_us\": " << total_time_us << ", ";
    std::cout << "\"gpu_kernel_time_us\": " << total_time_us << ", ";
    std::cout << "\"reduction_result\": " << final_result << ", ";
#ifdef IS_HETEROGENEOUS_AWARE
    std::cout << "\"openmp_max_threads\": " << omp_get_max_threads();
#else
    std::cout << "\"openmp_max_threads\": 1";
#endif
    std::cout << "}" << std::endl;

    return 0;
}