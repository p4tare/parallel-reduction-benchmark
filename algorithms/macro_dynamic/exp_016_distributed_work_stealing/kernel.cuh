#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <omp.h>

#define MAX_WORKERS 32

// Shared Atomic Counters for Work Stealing Queues
static std::atomic<size_t> worker_offsets[MAX_WORKERS];
static size_t worker_max_limits[MAX_WORKERS]; 
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
    
    // Configurable chunk size
    size_t CHUNK_SIZE = 1000000;
    if (CHUNK_SIZE > num_elements / total_workers) {
        CHUNK_SIZE = num_elements / (total_workers * 10);
        if (CHUNK_SIZE == 0) CHUNK_SIZE = 256;
    }

    double total_ms = 0.0;
    DATA_TYPE final_worker_val = 0;

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
            size_t elements_per_worker = num_elements / total_workers;
            size_t remainder = num_elements % total_workers;
            for (int i = 0; i < total_workers; ++i) {
                size_t my_el = elements_per_worker + (i == total_workers - 1 ? remainder : 0);
                worker_offsets[i].store(i * elements_per_worker);
                worker_max_limits[i] = i * elements_per_worker + my_el;
            }
            global_accumulated_result.store(0);
            if (is_trace_iter) printf("[TRACE] --- TOPOLOGY INITIALIZED. DISTRIBUTED WORK STEALING WITH HARDWARE-AWARE OMP --- \n");
        }
        #pragma omp barrier

        DATA_TYPE local_iteration_sum = 0;
        int chunks_own = 0;
        int chunks_stolen = 0;
        
        int target_queue = worker_id;
        auto start_time = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------
        // DYNAMIC WORK STEALING LOOP
        // ---------------------------------------------------------
        while (true) {
            size_t my_offset = worker_offsets[target_queue].fetch_add(CHUNK_SIZE);
            size_t my_elements = 0;
            
            if (my_offset < worker_max_limits[target_queue]) {
                my_elements = std::min(CHUNK_SIZE, worker_max_limits[target_queue] - my_offset);
                
                if (target_queue == worker_id) chunks_own++;
                else chunks_stolen++;
                
                // ==============================================================
                // CPU EXECUTION: HARDWARE-AWARE NESTED PARALLELISM
                // ==============================================================
                if (is_cpu_worker) {
                    // 1. Check how many cores Python 'taskset' left us
                    int num_procs_available = omp_get_num_procs(); 
                    int num_gpus = total_workers - 1;
                    
                    // 2. Calculate optimal threads for nested math
                    int nested_threads = num_procs_available;
                    if (dedicated_threads) {
                        nested_threads = num_procs_available - num_gpus;
                        if (nested_threads < 1) nested_threads = 1; // Failsafe
                    }

                    // 3. Telemetry Log (Only trigger on the very first chunk to avoid spam)
                    if (is_trace_iter && (chunks_own + chunks_stolen) == 1) {
                        printf("[TRACE] Worker 0 (CPU_MASTER) detected %d hardware cores available via taskset mask.\n", num_procs_available);
                        printf("[TRACE] Worker 0 (CPU_MASTER) dedicated_gpu_threads=%s -> Spawning %d nested OMP threads for chunk math!\n", 
                               dedicated_threads ? "TRUE" : "FALSE", nested_threads);
                    }

                    // 4. Execution
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
                    if (is_trace_iter && (chunks_own + chunks_stolen) <= 2) {
                        if (target_queue == worker_id) {
                            printf("[TRACE] Worker %d (GPU_%d) pulled chunk of %zu elements from LOCAL queue.\n", worker_id, gpu_id, my_elements);
                        } else {
                            printf("[TRACE] Worker %d (GPU_%d) STOLE chunk of %zu elements from Worker %d's queue.\n", worker_id, gpu_id, my_elements, target_queue);
                        }
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
                
            } else {
                // LOCAL QUEUE EMPTY -> STEAL PROTOCOL
                bool found = false;
                for (int i = 1; i < total_workers; ++i) {
                    int candidate = (worker_id + i) % total_workers;
                    if (worker_offsets[candidate].load() < worker_max_limits[candidate]) {
                        target_queue = candidate; 
                        found = true;
                        break;
                    }
                }
                if (!found) break; 
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
                printf("[TRACE] Worker %d (%s) finished iteration. Chunks processed - Own: %d, Stolen: %d. Sent local sum to RAM.\n", worker_id, role, chunks_own, chunks_stolen);
            } else {
                printf("[TRACE] Worker %d (GPU_%d) finished iteration. Chunks processed - Own: %d, Stolen: %d. Sent local sum to RAM.\n", worker_id, gpu_id, chunks_own, chunks_stolen);
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