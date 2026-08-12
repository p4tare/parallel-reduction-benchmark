#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <algorithm>

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

    // Uzywamy customAtomicAdd, o ktory dba nasz Wrapper!
    if (tid == 0) { customAtomicAdd(global_sum, sdata[0]); }
}

// HETEROGENEOUS STATIC EXECUTION WITH ASYNC PIPELINING
void execute_algorithm(DATA_TYPE* h_buffer, size_t num_elements, int repetitions, int warmup, 
                       int block_size, bool dedicated_threads, int worker_id, int total_workers, 
                       bool is_cpu_worker, int gpu_id, bool do_trace, double& out_time_us, double& out_result) {
    
    // 1. STATIC LOAD BALANCING
    double CPU_WEIGHT = 1.0;
    double GPU_WEIGHT = 15.0;
    double total_weight = 0.0;
    double offset_weight = 0.0;

    for (int i = 0; i < total_workers; ++i) {
        double w = (i == 0) ? CPU_WEIGHT : GPU_WEIGHT;
        total_weight += w;
        if (i < worker_id) offset_weight += w;
    }

    double my_weight = is_cpu_worker ? CPU_WEIGHT : GPU_WEIGHT;
    size_t my_offset = (size_t)((offset_weight / total_weight) * num_elements);
    size_t next_offset = (size_t)(((offset_weight + my_weight) / total_weight) * num_elements);
    if (worker_id == total_workers - 1) next_offset = num_elements; 
    
    size_t my_elements = next_offset - my_offset;
    if (my_elements == 0) return;

    double total_ms = 0.0;
    DATA_TYPE final_val = 0;

    // ==========================================
    // PATH A: CPU EXECUTION (Hardware-Aware Nested)
    // ==========================================
    if (is_cpu_worker) {
        int num_procs_available = omp_get_num_procs(); 
        int num_gpus = total_workers - 1;
        int nested_threads = dedicated_threads ? std::max(1, num_procs_available - num_gpus) : num_procs_available;

        for (int iter = 0; iter < warmup + repetitions; ++iter) {
            bool is_warmup = (iter < warmup);
            if (do_trace && iter == 0) {
                printf("[TRACE] Worker 0 (CPU_MASTER) assigned %zu elements (%.1f%% of data).\n", my_elements, (my_weight/total_weight)*100.0);
                printf("[TRACE] Worker 0 (CPU_MASTER) using %d nested OMP threads for static block.\n", nested_threads);
            }

            auto start_cpu = std::chrono::high_resolution_clock::now();
            DATA_TYPE sum = 0;
            
            #pragma omp parallel for simd num_threads(nested_threads) reduction(+:sum)
            for (size_t j = 0; j < my_elements; ++j) { 
                sum += h_buffer[my_offset + j]; 
            }
            
            auto stop_cpu = std::chrono::high_resolution_clock::now();
            
            if (!is_warmup) {
                std::chrono::duration<double, std::milli> ms = stop_cpu - start_cpu;
                total_ms += ms.count();
            }
            final_val = sum;
        }
        
        out_time_us = (total_ms * 1000.0) / repetitions;
        out_result = static_cast<double>(final_val);
        return;
    }

    // ==========================================
    // PATH B: GPU EXECUTION (Async Pipelining)
    // ==========================================
    
    // PIPELINE CONFIGURATION
    const int NUM_STREAMS = 4;   // 4 Concurrent CUDA Streams
    const int NUM_CHUNKS = 16;   // Divide data into 16 pipelined chunks
    
    size_t chunk_size = (my_elements + NUM_CHUNKS - 1) / NUM_CHUNKS;
    size_t shared_mem_size = block_size * sizeof(DATA_TYPE);

    // MAGIC TRICK: Pin the host memory on-the-fly to allow True Async DMA Transfers
    cudaError_t pin_status = cudaHostRegister(h_buffer + my_offset, my_elements * sizeof(DATA_TYPE), cudaHostRegisterDefault);
    bool is_pinned = (pin_status == cudaSuccess);

    cudaStream_t streams[NUM_STREAMS];
    DATA_TYPE* d_inputs[NUM_STREAMS];
    DATA_TYPE* d_results[NUM_STREAMS];
    
    // Host results array must also be pinned for async D2H
    DATA_TYPE* h_results;
    CUDA_CHECK(cudaHostAlloc(&h_results, NUM_CHUNKS * sizeof(DATA_TYPE), cudaHostAllocDefault));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_inputs[i], chunk_size * sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMalloc(&d_results[i], sizeof(DATA_TYPE)));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int iter = 0; iter < warmup + repetitions; ++iter) {
        bool is_warmup = (iter < warmup);
        
        if (do_trace && iter == 0) {
            printf("[TRACE] Worker %d (GPU_%d) assigned %zu elements (%.1f%% of data).\n", worker_id, gpu_id, my_elements, (my_weight/total_weight)*100.0);
            if (is_pinned) {
                printf("[TRACE] Worker %d (GPU_%d) successfully locked RAM via cudaHostRegister. TRUE ASYNC ENABLED.\n", worker_id, gpu_id);
            } else {
                printf("[TRACE] Worker %d (GPU_%d) cudaHostRegister failed. Falling back to synchronous driver staging.\n", worker_id, gpu_id);
            }
            printf("[TRACE] Worker %d (GPU_%d) initiating %d-Stream Pipeline across %d chunks...\n", worker_id, gpu_id, NUM_STREAMS, NUM_CHUNKS);
        }

        if (!is_warmup) CUDA_CHECK(cudaEventRecord(start));

        // -------------------------------------------------------------
        // ASYNC PIPELINE DISPATCH LOOP
        // -------------------------------------------------------------
        for (int c = 0; c < NUM_CHUNKS; ++c) {
            int s = c % NUM_STREAMS; // Round-robin stream assignment
            
            size_t c_offset = my_offset + (c * chunk_size);
            size_t c_elements = std::min(chunk_size, my_elements - (c * chunk_size));
            if (c_elements <= 0) break;

            int current_grid = (c_elements + block_size - 1) / block_size;
            if (current_grid > 1024) current_grid = 1024;

            // Zlecanie zadan w tle (Enqueue) do strumienia s
            DATA_TYPE zero = 0;
            CUDA_CHECK(cudaMemcpyAsync(d_results[s], &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(d_inputs[s], h_buffer + c_offset, c_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[s]));
            
            reduce_kernel<<<current_grid, block_size, shared_mem_size, streams[s]>>>(d_inputs[s], c_elements, d_results[s]);
            
            CUDA_CHECK(cudaMemcpyAsync(&h_results[c], d_results[s], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[s]));

            if (do_trace && iter == 0 && c < NUM_STREAMS) {
                printf("[TRACE] Worker %d (GPU_%d) enqueued Chunk %d into Stream %d [H2D -> Kernel -> D2H]\n", worker_id, gpu_id, c, s);
            }
        }
        
        // Wait for all streams in the pipeline to drain and finish
        CUDA_CHECK(cudaDeviceSynchronize());

        if (!is_warmup) {
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }

        // Host-side accumulation of the pipelined results
        DATA_TYPE iter_val = 0;
        for (int c = 0; c < NUM_CHUNKS; ++c) {
            if (c * chunk_size < my_elements) iter_val += h_results[c];
        }
        final_val = iter_val;
    }

    out_time_us = (total_ms * 1000.0) / repetitions;
    out_result = static_cast<double>(final_val);

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_inputs[i]));
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_results));
    
    if (is_pinned) { cudaHostUnregister(h_buffer + my_offset); }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}