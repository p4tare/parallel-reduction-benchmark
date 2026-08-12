// algorithms/macro_dynamic/exp_013_v3_hierarchical/kernel.cuh
#ifndef EXP_013_V3_HIERARCHICAL_CUH
#define EXP_013_V3_HIERARCHICAL_CUH

#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <iostream>

#define MAX_WORKERS 32
#define NUM_STREAMS 2

static size_t worker_offsets[MAX_WORKERS];
static size_t worker_sizes[MAX_WORKERS];
static double worker_benchmark_times[MAX_WORKERS];
static int gpu_id_map[MAX_WORKERS];

static DATA_TYPE* d_inputs[MAX_WORKERS][NUM_STREAMS];
static DATA_TYPE* d_outputs[MAX_WORKERS][NUM_STREAMS];
static cudaStream_t streams[MAX_WORKERS][NUM_STREAMS];

// Dedicated VRAM buffer for GPU P2P result pooling
static DATA_TYPE* d_gpu_p2p_pool[MAX_WORKERS];

template <typename T>
__global__ void reduce_kernel(const T* input, size_t n, T* global_sum) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = 0;
    while (i < n) { sum += input[i]; i += blockDim.x * gridDim.x; }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { customAtomicAdd(global_sum, sdata[0]); }
}

template <typename T>
__global__ void p2p_merge_kernel(const T* input_pool, size_t num_gpus, T* final_sum) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        T sum = 0;
        for (int i = 0; i < num_gpus; ++i) { sum += input_pool[i]; }
        *final_sum = sum;
    }
}

void algorithm_setup(DATA_TYPE* h_buffer, size_t num_elements, int block_size, bool dedicated_threads, 
                     int worker_id, int total_workers, bool is_cpu_worker, int gpu_id, bool do_trace) {
    
    gpu_id_map[worker_id] = is_cpu_worker ? -1 : gpu_id;
    
    // Enable Peer-to-Peer access for Hierarchical GPU phase
    if (!is_cpu_worker) {
        for (int i = 0; i < total_workers; ++i) {
            if (i != worker_id && !is_cpu_worker && gpu_id_map[i] != -1) {
                cudaError_t err = cudaDeviceEnablePeerAccess(gpu_id_map[i], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            }
        }
    }
    
    size_t benchmark_size = 100000;
    if (benchmark_size > num_elements) benchmark_size = num_elements;
    
    auto start_bench = std::chrono::high_resolution_clock::now();
    if (is_cpu_worker) {
        DATA_TYPE dummy_sum = 0;
        #pragma omp parallel for simd reduction(+:dummy_sum)
        for (size_t i = 0; i < benchmark_size; ++i) { dummy_sum += h_buffer[i]; }
    } else {
        DATA_TYPE* d_bench_in; DATA_TYPE* d_bench_out; cudaStream_t bench_stream;
        cudaStreamCreate(&bench_stream);
        cudaMalloc(&d_bench_in, benchmark_size * sizeof(DATA_TYPE));
        cudaMalloc(&d_bench_out, sizeof(DATA_TYPE));
        
        DATA_TYPE zero = 0;
        cudaMemcpyAsync(d_bench_out, &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, bench_stream);
        cudaMemcpyAsync(d_bench_in, h_buffer, benchmark_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, bench_stream);
        int grid = (benchmark_size + block_size - 1) / block_size;
        if (grid > 1024) grid = 1024;
        reduce_kernel<<<grid, block_size, block_size * sizeof(DATA_TYPE), bench_stream>>>(d_bench_in, benchmark_size, d_bench_out);
        
        DATA_TYPE res;
        cudaMemcpyAsync(&res, d_bench_out, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, bench_stream);
        cudaStreamSynchronize(bench_stream);
        cudaFree(d_bench_in); cudaFree(d_bench_out); cudaStreamDestroy(bench_stream);
    }
    auto end_bench = std::chrono::high_resolution_clock::now();
    worker_benchmark_times[worker_id] = std::chrono::duration<double>(end_bench - start_bench).count();
    
    #pragma omp barrier

    if (worker_id == 0) {
        double total_weight = 0.0;
        double weights[MAX_WORKERS];
        for (int i = 0; i < total_workers; ++i) {
            weights[i] = 1.0 / worker_benchmark_times[i];
            total_weight += weights[i];
        }
        size_t current_offset = 0;
        for (int i = 0; i < total_workers; ++i) {
            worker_offsets[i] = current_offset;
            worker_sizes[i] = (i == total_workers - 1) ? (num_elements - current_offset) : (size_t)((weights[i] / total_weight) * num_elements);
            current_offset += worker_sizes[i];
        }
        if (do_trace) printf("[TRACE] EXP013_V3: Hierarchical Topology Generated.\n");
    }
    #pragma omp barrier

    if (!is_cpu_worker && worker_sizes[worker_id] > 0) {
        size_t stream_chunk_size = (worker_sizes[worker_id] + NUM_STREAMS - 1) / NUM_STREAMS;
        for(int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaMalloc(&d_inputs[worker_id][s], stream_chunk_size * sizeof(DATA_TYPE)));
            CUDA_CHECK(cudaMalloc(&d_outputs[worker_id][s], sizeof(DATA_TYPE)));
            CUDA_CHECK(cudaStreamCreate(&streams[worker_id][s]));
        }
        CUDA_CHECK(cudaMalloc(&d_gpu_p2p_pool[worker_id], total_workers * sizeof(DATA_TYPE)));
        cudaHostRegister(h_buffer + worker_offsets[worker_id], worker_sizes[worker_id] * sizeof(DATA_TYPE), cudaHostRegisterDefault);
    }
}

void algorithm_execute(DATA_TYPE* h_buffer, size_t num_elements, int block_size, bool dedicated_threads, 
                       int worker_id, int total_workers, bool is_cpu_worker, int gpu_id, bool do_trace, 
                       double& out_result) {
    out_result = 0.0;
    size_t my_size = worker_sizes[worker_id];
    size_t my_offset = worker_offsets[worker_id];
    
    // Find the leading nodes for hierarchical reduction
    int first_cpu_worker = -1;
    int first_gpu_worker = -1;
    for (int i = 0; i < total_workers; ++i) {
        if (gpu_id_map[i] == -1 && first_cpu_worker == -1) first_cpu_worker = i;
        if (gpu_id_map[i] != -1 && first_gpu_worker == -1) first_gpu_worker = i;
    }

    if (is_cpu_worker && my_size > 0) {
        DATA_TYPE chunk_sum = 0;
        int nested_threads = dedicated_threads ? std::max(1, omp_get_num_procs() - (total_workers - 1)) : omp_get_num_procs();
        #pragma omp parallel for simd num_threads(nested_threads) reduction(+:chunk_sum)
        for (size_t j = 0; j < my_size; ++j) { chunk_sum += h_buffer[my_offset + j]; }
        out_result = static_cast<double>(chunk_sum); 
        // Note: The OpenMP wrapper `#pragma omp critical` automatically handles Phase 4 CPU Subsystem Merging inherently.
    } else if (!is_cpu_worker && my_size > 0) {
        DATA_TYPE h_gpu_outputs[NUM_STREAMS] = {0};
        size_t stream_chunk_size = (my_size + NUM_STREAMS - 1) / NUM_STREAMS;

        for(int s = 0; s < NUM_STREAMS; ++s) {
            size_t c_offset = my_offset + (s * stream_chunk_size);
            size_t c_elements = std::min(stream_chunk_size, my_size - (s * stream_chunk_size));
            if (c_elements <= 0) break;

            DATA_TYPE zero = 0;
            CUDA_CHECK(cudaMemcpyAsync(d_outputs[worker_id][s], &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id][s]));
            CUDA_CHECK(cudaMemcpyAsync(d_inputs[worker_id][s], h_buffer + c_offset, c_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id][s]));

            int grid = (c_elements + block_size - 1) / block_size;
            if (grid > 1024) grid = 1024;
            reduce_kernel<<<grid, block_size, block_size * sizeof(DATA_TYPE), streams[worker_id][s]>>>(d_inputs[worker_id][s], c_elements, d_outputs[worker_id][s]);
            CUDA_CHECK(cudaMemcpyAsync(&h_gpu_outputs[s], d_outputs[worker_id][s], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[worker_id][s]));
        }

        DATA_TYPE combined_local_sum = 0;
        for(int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[worker_id][s]));
            combined_local_sum += h_gpu_outputs[s];
        }

        // PHASE 4: P2P VRAM-to-VRAM Result Merging 
        if (worker_id != first_gpu_worker) {
            // Sub-GPUs send their result to the Lead GPU via NVLink/PCIe
            CUDA_CHECK(cudaMemcpyPeerAsync(d_gpu_p2p_pool[first_gpu_worker] + worker_id, gpu_id_map[first_gpu_worker], 
                                           &combined_local_sum, gpu_id, sizeof(DATA_TYPE), streams[worker_id][0]));
            CUDA_CHECK(cudaStreamSynchronize(streams[worker_id][0]));
            out_result = 0.0; // Handled by leader
        } else {
            // Lead GPU writes its own result to the pool
            CUDA_CHECK(cudaMemcpyAsync(d_gpu_p2p_pool[worker_id] + worker_id, &combined_local_sum, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id][0]));
        }
    }

    #pragma omp barrier // Synchronize all workers before the Master GPU synthesizes the subsystem sum

    if (!is_cpu_worker && worker_id == first_gpu_worker && my_size > 0) {
        DATA_TYPE zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_outputs[worker_id][0], &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id][0]));
        
        // Final subsystem merge entirely in VRAM
        p2p_merge_kernel<<<1, 1, 0, streams[worker_id][0]>>>(d_gpu_p2p_pool[worker_id], total_workers, d_outputs[worker_id][0]);
        
        DATA_TYPE gpu_subsystem_total = 0;
        CUDA_CHECK(cudaMemcpyAsync(&gpu_subsystem_total, d_outputs[worker_id][0], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[worker_id][0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[worker_id][0]));
        out_result = static_cast<double>(gpu_subsystem_total);
    }
}

void algorithm_teardown(int worker_id, bool is_cpu_worker, int gpu_id) {
    if (!is_cpu_worker && worker_sizes[worker_id] > 0) {
        cudaHostUnregister(worker_offsets[worker_id] + (DATA_TYPE*)0); 
        for(int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaFree(d_inputs[worker_id][s]));
            CUDA_CHECK(cudaFree(d_outputs[worker_id][s]));
            CUDA_CHECK(cudaStreamDestroy(streams[worker_id][s]));
        }
        CUDA_CHECK(cudaFree(d_gpu_p2p_pool[worker_id]));
    }
}
#endif