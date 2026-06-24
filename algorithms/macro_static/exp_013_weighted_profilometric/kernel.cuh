#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// GPU KERNEL 
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
                       bool is_cpu_worker, int gpu_id, bool do_trace, double& out_time_us, double& out_result) {
    
    double CPU_WEIGHT = 1.0;
    double GPU_WEIGHT = 15.0;
    
    double total_weight = 0.0;
    double offset_weight = 0.0;

    for (int i = 0; i < total_workers; ++i) {
        double w = (i == 0) ? CPU_WEIGHT : GPU_WEIGHT;
        total_weight += w;
        if (i < worker_id) {
            offset_weight += w;
        }
    }

    double my_weight = is_cpu_worker ? CPU_WEIGHT : GPU_WEIGHT;
    
    size_t my_offset = (size_t)((offset_weight / total_weight) * num_elements);
    size_t next_offset = (size_t)(((offset_weight + my_weight) / total_weight) * num_elements);
    
    if (worker_id == total_workers - 1) {
        next_offset = num_elements; 
    }
    
    size_t my_elements = next_offset - my_offset;

    if (my_elements == 0) return;

    double total_ms = 0.0;
    DATA_TYPE final_val = 0;

    // ==========================================
    // PATH A: CPU WORKER EXECUTION
    // ==========================================
    if (is_cpu_worker) {
        for (int i = 0; i < warmup; ++i) {
            if (do_trace && i == 0) {
                printf("[TRACE] Worker %d (CPU) assigned weight %.1f (%.1f%% of data). Elements: %zu\n", 
                       worker_id, my_weight, (my_weight/total_weight)*100.0, my_elements);
                printf("[TRACE] Worker %d (CPU) reading %zu elements directly from main RAM (Offset: %zu)...\n", worker_id, my_elements, my_offset);
                printf("[TRACE] Worker %d (CPU) executing SIMD AVX vector loop...\n", worker_id);
            }

            DATA_TYPE sum = 0;
            for (size_t j = 0; j < my_elements; ++j) { sum += h_buffer[my_offset + j]; }
            final_val = sum; 
            
            if (do_trace && i == 0) {
                printf("[TRACE] Worker %d (CPU) completed reduction. Partial sum stored in RAM: %.2f\n", worker_id, (double)final_val);
            }
        }

        for (int i = 0; i < repetitions; ++i) {
            auto start_cpu = std::chrono::high_resolution_clock::now();
            DATA_TYPE sum = 0;
            for (size_t j = 0; j < my_elements; ++j) { sum += h_buffer[my_offset + j]; }
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

    for (int i = 0; i < warmup; ++i) {
        if (do_trace && i == 0) {
            printf("[TRACE] Worker %d (GPU_%d) assigned weight %.1f (%.1f%% of data). Elements: %zu\n", 
                   worker_id, gpu_id, my_weight, (my_weight/total_weight)*100.0, my_elements);
            printf("[TRACE] Worker %d (GPU_%d) initiating PCIe transfer (Host RAM -> GPU VRAM) for offset %zu...\n", worker_id, gpu_id, my_offset);
            printf("[TRACE] Worker %d (GPU_%d) launching reduce_kernel (Grid: %d, Block: %d)...\n", worker_id, gpu_id, grid_size, block_size);
        }

        DATA_TYPE zero = 0;
        CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, h_buffer + my_offset, my_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
        reduce_kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, my_elements, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Fetch result during warmup to simulate full data lifecycle for trace
        DATA_TYPE temp_result = 0;
        CUDA_CHECK(cudaMemcpy(&temp_result, d_result, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
        
        if (do_trace && i == 0) {
            printf("[TRACE] Worker %d (GPU_%d) reduction complete. Initiating PCIe transfer (GPU VRAM -> Host RAM)...\n", worker_id, gpu_id);
            printf("[TRACE] Worker %d (GPU_%d) partial result fetched: %.2f\n", worker_id, gpu_id, (double)temp_result);
        }
    }

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