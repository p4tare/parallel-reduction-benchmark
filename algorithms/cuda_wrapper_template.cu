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

// =====================================================================
// UNIWERSALNE OPERACJE ATOMOWE (WIDOCZNE DLA KAZDEGO ALGORYTMU)
// =====================================================================
__device__ inline void atomicAdd(float2* address, float2 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
}
__device__ inline void atomicAdd(float3* address, float3 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
    atomicAdd(&(address->z), val.z);
}
__device__ inline void atomicAdd(float4* address, float4 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
    atomicAdd(&(address->z), val.z);
    atomicAdd(&(address->w), val.w);
}

// =====================================================================
// WSTRZYKNIÊCIE ALGORYTMU (Oczekujemy: setup, execute, teardown)
// =====================================================================
#include "kernel.cuh"

std::vector<int> parse_gpu_list(std::string gpu_str) {
    std::vector<int> gpus;
    gpu_str.erase(std::remove(gpu_str.begin(), gpu_str.end(), '['), gpu_str.end());
    gpu_str.erase(std::remove(gpu_str.begin(), gpu_str.end(), ']'), gpu_str.end());
    gpu_str.erase(std::remove(gpu_str.begin(), gpu_str.end(), ' '), gpu_str.end());
    
    std::stringstream ss(gpu_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) gpus.push_back(std::stoi(token));
    }
    if (gpus.empty()) gpus.push_back(0); 
    return gpus;
}

int main(int argc, char** argv) {
    std::string data_path = "";
    std::string gpus_raw = "0";
    int reps = 10;
    int warmup = 3;
    bool dedicated_threads = false;
    int block_size = 256;
    bool do_trace = false;

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

    // Wektory do zapisywania pojedynczych iteracji
    std::vector<double> iter_max_times(reps, 0.0);
    std::vector<double> iter_sum_results(reps, 0.0);
    
    size_t elements_read = 0;
    int total_workers = 1 + num_gpus; 
    omp_set_num_threads(total_workers);

    while (elements_read < total_elements) {
        size_t elements_to_read = std::min(current_chunk_size, total_elements - elements_read);
        fread(h_buffer, sizeof(DATA_TYPE), elements_to_read, file);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            bool is_cpu_worker = (thread_id == 0);
            int my_gpu_id = (!is_cpu_worker) ? target_gpus[thread_id - 1] : -1;
            
            if (!is_cpu_worker) { CUDA_CHECK(cudaSetDevice(my_gpu_id)); }
            
            // 1. Inicjalizacja sprzêtowa w izolacji
            algorithm_setup(h_buffer, elements_to_read, block_size, dedicated_threads, thread_id, total_workers, is_cpu_worker, my_gpu_id, do_trace);
            
            // 2. Faza rozgrzewki sprzêtu (Warmup)
            for (int w = 0; w < warmup; ++w) {
                double dummy = 0;
                algorithm_execute(h_buffer, elements_to_read, block_size, dedicated_threads, thread_id, total_workers, is_cpu_worker, my_gpu_id, false, dummy);
                #pragma omp barrier
            }
            
            // 3. Faza Ostrego Pomiaru (Reps)
            for (int r = 0; r < reps; ++r) {
                double res = 0;
                bool trace_this_iter = (do_trace && r == 0); // Trace only the very first measured iteration
                
                #pragma omp barrier // Synchronize all workers before the stopwatch starts
                auto start_time = std::chrono::high_resolution_clock::now();
                
                algorithm_execute(h_buffer, elements_to_read, block_size, dedicated_threads, thread_id, total_workers, is_cpu_worker, my_gpu_id, trace_this_iter, res);
                
                auto stop_time = std::chrono::high_resolution_clock::now();
                double time_us = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
                
                #pragma omp critical
                {
                    iter_sum_results[r] += res;
                    if (time_us > iter_max_times[r]) {
                        iter_max_times[r] = time_us;
                    }
                }
                #pragma omp barrier
            }
            
            // 4. Sprz¹tanie
            algorithm_teardown(thread_id, is_cpu_worker, my_gpu_id);
        }

        elements_read += elements_to_read;
    }

    fclose(file);
    delete[] h_buffer;

    // JSON Dump: Format tablicy [ {}, {}, ... ]
    std::cout << "\n[";
    for(int r = 0; r < reps; ++r) {
        std::cout << "{\"iteration\": " << (r + 1) 
                  << ", \"cpp_cpu_time_us\": " << iter_max_times[r]
                  << ", \"gpu_kernel_time_us\": " << iter_max_times[r]
                  << ", \"reduction_result\": " << iter_sum_results[r] 
                  << ", \"openmp_max_threads\": " << omp_get_max_threads() << "}";
        if(r < reps - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    return 0;
}