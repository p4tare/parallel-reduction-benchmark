#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <atomic>

// Shared Atomic Counters
static std::atomic<size_t> global_task_offset{0};
static std::atomic<double> global_accumulated_result{0};

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

// HETEROGENEOUS DYNAMIC EXECUTION
void execute_algorithm(DATA_TYPE* h_buffer, size_t num_elements, int repetitions, int warmup, 
                       int block_size, bool dedicated_threads, int worker_id, int total_workers, 
                       bool is_cpu_worker, int gpu_id, bool do_trace, double& out_time_us, double& out_result) {
    
    // 1. DYNAMIC CHUNK SIZING (Slice workload into ~100 chunks)
    size_t CHUNK_SIZE = num_elements / 100; 
    
    // Safety boundaries
    if (CHUNK_SIZE < 256) CHUNK_SIZE = 256; 
    if (num_elements < 256) CHUNK_SIZE = num_elements;
    
    double total_ms = 0.0;
    DATA_TYPE final_worker_val = 0;

    // Allocate GPU memory once based on dynamically calculated CHUNK_SIZE
    DATA_TYPE* d_input = nullptr;
    DATA_TYPE* d_result = nullptr;
    
    if (!is_cpu_worker) {
        CUDA_CHECK(cudaMalloc(&d_input, CHUNK_SIZE * sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(DATA_TYPE)));
    }

    int grid_size = (CHUNK_SIZE + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024; 
    size_t shared_mem_size = block_size * sizeof(DATA_TYPE);

    for (int iter = 0; iter < warmup + repetitions; ++iter) {
        bool is_warmup = (iter < warmup);
        bool is_trace_iter = (do_trace && iter == 0);

        #pragma omp barrier
        if (worker_id == 0) {
            global_task_offset.store(0);
            global_accumulated_result.store(0);
            if (is_trace_iter) printf("[TRACE] --- QUEUE RESET FOR NEW ITERATION. DYNAMIC CHUNK SIZE: %zu ---\n", CHUNK_SIZE);
        }
        #pragma omp barrier

        DATA_TYPE local_iteration_sum = 0;
        int chunks_processed = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------
        // DYNAMIC WORK STEALING LOOP
        // ---------------------------------------------------------
        while (true) {
            size_t my_offset = global_task_offset.fetch_add(CHUNK_SIZE);
            if (my_offset >= num_elements) break; 

            size_t my_elements = std::min(CHUNK_SIZE, num_elements - my_offset);
            chunks_processed++;

            if (is_trace_iter && chunks_processed <= 2) {
                if (is_cpu_worker) {
                    printf("[TRACE] Worker %d (CPU) dynamically pulled chunk of %zu elements at offset %zu\n", worker_id, my_elements, my_offset);
                } else {
                    printf("[TRACE] Worker %d (GPU_%d) dynamically pulled chunk of %zu elements at offset %zu\n", worker_id, gpu_id, my_elements, my_offset);
                }
            }

            if (is_cpu_worker) {
                DATA_TYPE chunk_sum = 0;
                for (size_t j = 0; j < my_elements; ++j) {
                    chunk_sum += h_buffer[my_offset + j];
                }
                local_iteration_sum += chunk_sum;
            } else {
                DATA_TYPE zero = 0;
                CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_input, h_buffer + my_offset, my_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
                
                int current_grid = (my_elements + block_size - 1) / block_size;
                if (current_grid > 1024) current_grid = 1024;
                
                reduce_kernel<<<current_grid, block_size, shared_mem_size>>>(d_input, my_elements, d_result);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                DATA_TYPE chunk_sum = 0;
                CUDA_CHECK(cudaMemcpy(&chunk_sum, d_result, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
                local_iteration_sum += chunk_sum;
            }
        }
        // ---------------------------------------------------------

        auto stop_time = std::chrono::high_resolution_clock::now();

        #pragma omp critical
        {
            double current_global = global_accumulated_result.load();
            global_accumulated_result.store(current_global + (double)local_iteration_sum);
        }

        // Improved Summary Log
        if (is_trace_iter) {
            if (is_cpu_worker) {
                printf("[TRACE] Worker %d (CPU) finished iteration. Total chunks processed: %d. Accumulated local sum sent to global RAM counter.\n", worker_id, chunks_processed);
            } else {
                printf("[TRACE] Worker %d (GPU_%d) finished iteration. Total chunks processed: %d. Accumulated local sum sent to global RAM counter.\n", worker_id, gpu_id, chunks_processed);
            }
        }

        #pragma omp barrier

        if (!is_warmup) {
            std::chrono::duration<double, std::milli> ms = stop_time - start_time;
            total_ms += ms.count();
        }

        if (worker_id == 0 && !is_warmup) {
            final_worker_val = (DATA_TYPE)global_accumulated_result.load();
        }
    }

    if (!is_cpu_worker) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_result));
    }

    out_time_us = (total_ms * 1000.0) / repetitions;
    out_result = static_cast<double>(final_worker_val);
}