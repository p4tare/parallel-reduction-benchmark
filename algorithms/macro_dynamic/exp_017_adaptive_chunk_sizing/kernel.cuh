#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <atomic>
#include <omp.h>

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
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }

    if (tid == 0) { customAtomicAdd(global_sum, sdata[0]); }
}

// HETEROGENEOUS DYNAMIC EXECUTION
void execute_algorithm(DATA_TYPE* h_buffer, size_t num_elements, int repetitions, int warmup, 
                       int block_size, bool dedicated_threads, int worker_id, int total_workers, 
                       bool is_cpu_worker, int gpu_id, bool do_trace, double& out_time_us, double& out_result) {
    
    double total_ms = 0.0;
    DATA_TYPE final_worker_val = 0;

    size_t MAX_ALLOC_SIZE = num_elements; 
    if (MAX_ALLOC_SIZE > 50000000) MAX_ALLOC_SIZE = 50000000; 

    DATA_TYPE* d_input = nullptr;
    DATA_TYPE* d_result = nullptr;
    
    if (!is_cpu_worker) {
        CUDA_CHECK(cudaMalloc(&d_input, MAX_ALLOC_SIZE * sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(DATA_TYPE)));
    }

    size_t shared_mem_size = block_size * sizeof(DATA_TYPE);

    for (int iter = 0; iter < warmup + repetitions; ++iter) {
        bool is_warmup = (iter < warmup);
        bool is_trace_iter = (do_trace && iter == 0);

        #pragma omp barrier
        if (worker_id == 0) {
            global_task_offset.store(0);
            global_accumulated_result.store(0);
            if (is_trace_iter) printf("[TRACE] --- QUEUE RESET. HARDWARE-AWARE ADAPTIVE CHUNK SIZING INITIALIZED ---\n");
        }
        #pragma omp barrier

        DATA_TYPE local_iteration_sum = 0;
        int chunks_processed = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------
        // DYNAMIC ADAPTIVE LOOP
        // ---------------------------------------------------------
        while (true) {
            size_t current_offset = global_task_offset.load();
            size_t my_elements = 0;
            
            while (true) {
                if (current_offset >= num_elements) break;
                
                size_t remaining = num_elements - current_offset;
                
                // Zmniejszanie paczki logarytmicznie (Adaptive)
                size_t adaptive_chunk = remaining / (2 * total_workers);
                
                if (adaptive_chunk < 1000) adaptive_chunk = remaining; 
                if (adaptive_chunk > MAX_ALLOC_SIZE) adaptive_chunk = MAX_ALLOC_SIZE;

                if (global_task_offset.compare_exchange_weak(current_offset, current_offset + adaptive_chunk)) {
                    my_elements = adaptive_chunk;
                    break;
                }
            }
            if (my_elements == 0) break;
            
            size_t my_offset = current_offset;
            chunks_processed++;

            // ==============================================================
            // CPU EXECUTION: HARDWARE-AWARE NESTED PARALLELISM
            // ==============================================================
            if (is_cpu_worker) {
                int num_procs_available = omp_get_num_procs(); 
                int num_gpus = total_workers - 1;
                
                int nested_threads = num_procs_available;
                if (dedicated_threads) {
                    nested_threads = num_procs_available - num_gpus;
                    if (nested_threads < 1) nested_threads = 1; 
                }

                if (is_trace_iter && chunks_processed <= 2) {
                    size_t remaining = num_elements - current_offset;
                    printf("[TRACE] Worker 0 (CPU_MASTER) evaluated queue. Remaining: %zu. Adapted chunk: %zu.\n", remaining, my_elements);
                    printf("[TRACE] Worker 0 (CPU_MASTER) Spawning %d nested OMP threads for chunk math!\n", nested_threads);
                }

                DATA_TYPE chunk_sum = 0;
                
                #pragma omp parallel for simd num_threads(nested_threads) reduction(+:chunk_sum)
                for (size_t j = 0; j < my_elements; ++j) { 
                    chunk_sum += h_buffer[my_offset + j]; 
                }
                local_iteration_sum += chunk_sum;
            } 
            // ==============================================================
            // GPU EXECUTION
            // ==============================================================
            else {
                if (is_trace_iter && chunks_processed <= 2) {
                    size_t remaining = num_elements - current_offset;
                    printf("[TRACE] Worker %d (GPU_%d) evaluated queue. Remaining: %zu. Adapted chunk: %zu.\n", worker_id, gpu_id, remaining, my_elements);
                }

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

        if (is_trace_iter) {
            const char* role = is_cpu_worker ? "CPU_MASTER" : "GPU";
            if (is_cpu_worker) {
                printf("[TRACE] Worker %d (%s) finished iteration. Processed ADAPTIVE chunks: %d. Sent local sum to RAM.\n", worker_id, role, chunks_processed);
            } else {
                printf("[TRACE] Worker %d (GPU_%d) finished iteration. Processed ADAPTIVE chunks: %d. Sent local sum to RAM.\n", worker_id, gpu_id, chunks_processed);
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