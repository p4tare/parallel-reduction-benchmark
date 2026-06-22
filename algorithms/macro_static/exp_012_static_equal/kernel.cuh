#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// GPU KERNEL (Worker Logic)
template <typename T>
__global__ void reduce_kernel(const T* input, size_t n, T* global_sum) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;
    while (i < n) {
        sum += input[i];
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        customAtomicAdd(global_sum, sdata[0]);
    }
}

// HETEROGENEOUS EXECUTION SIGNATURE
void execute_algorithm(DATA_TYPE* h_buffer, size_t num_elements, int repetitions, int warmup, 
                       int block_size, bool dedicated_threads, int worker_id, int total_workers, 
                       bool is_cpu_worker, int gpu_id, double& out_time_us, double& out_result) {
    
    // 1. CALCULATE EXACT WORKLOAD FOR THIS WORKER (CPU OR GPU)
    size_t elements_per_worker = num_elements / total_workers;
    size_t remainder = num_elements % total_workers;
    
    size_t my_offset = worker_id * elements_per_worker;
    size_t my_elements = elements_per_worker;
    
    // The last worker cleans up any remainder elements
    if (worker_id == total_workers - 1) {
        my_elements += remainder; 
    }

    if (my_elements == 0) {
        out_time_us = 0.0;
        out_result = 0.0;
        return;
    }

    double total_ms = 0.0;
    DATA_TYPE final_val = 0;

    // ==========================================
    // PATH A: CPU WORKER EXECUTION
    // ==========================================
    if (is_cpu_worker) {
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            DATA_TYPE sum = 0;
            for (size_t j = 0; j < my_elements; ++j) {
                sum += h_buffer[my_offset + j];
            }
            final_val = sum; 
        }

        // Measurement
        for (int i = 0; i < repetitions; ++i) {
            auto start_cpu = std::chrono::high_resolution_clock::now();
            
            DATA_TYPE sum = 0;
            // Native CPU Loop - Compiler will auto-vectorize this to SIMD AVX
            for (size_t j = 0; j < my_elements; ++j) {
                sum += h_buffer[my_offset + j];
            }
            
            auto stop_cpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = stop_cpu - start_cpu;
            
            total_ms += ms.count();
            final_val = sum;
        }
        
        out_time_us = (total_ms * 1000.0) / repetitions;
        out_result = static_cast<double>(final_val);
        return;
    }

    // ==========================================
    // PATH B: GPU WORKER EXECUTION
    // ==========================================
    DATA_TYPE* d_input = nullptr;
    DATA_TYPE* d_result = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, my_elements * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(DATA_TYPE)));

    int grid_size = (my_elements + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024; 
    
    size_t shared_mem_size = block_size * sizeof(DATA_TYPE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        DATA_TYPE zero = 0;
        CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, h_buffer + my_offset, my_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
        reduce_kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, my_elements, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measurement
    for (int i = 0; i < repetitions; ++i) {
        DATA_TYPE zero = 0;
        CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_input, h_buffer + my_offset, my_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
        reduce_kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, my_elements, d_result);
        CUDA_CHECK(cudaMemcpy(&final_val, d_result, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    out_time_us = (total_ms * 1000.0) / repetitions;
    out_result = static_cast<double>(final_val);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}