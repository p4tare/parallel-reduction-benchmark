// algorithms/macro_dynamic/exp_013_v1_basic/kernel.cuh
#ifndef EXP_013_V1_BASIC_CUH
#define EXP_013_V1_BASIC_CUH

#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <iostream>

#define MAX_WORKERS 32

// Global state for Profilometric Load Balancing
static size_t worker_offsets[MAX_WORKERS];
static size_t worker_sizes[MAX_WORKERS];
static double worker_benchmark_times[MAX_WORKERS];

static DATA_TYPE* d_inputs[MAX_WORKERS];
static DATA_TYPE* d_outputs[MAX_WORKERS];
static cudaStream_t streams[MAX_WORKERS];

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

void algorithm_setup(DATA_TYPE* h_buffer, size_t num_elements, int block_size, bool dedicated_threads, 
                     int worker_id, int total_workers, bool is_cpu_worker, int gpu_id, bool do_trace) {
    
    // PHASE 1: Profiling (Benchmark)
    size_t benchmark_size = 100000; // Small sample for speed test
    if (benchmark_size > num_elements) benchmark_size = num_elements;
    
    auto start_bench = std::chrono::high_resolution_clock::now();
    
    if (is_cpu_worker) {
        DATA_TYPE dummy_sum = 0;
        #pragma omp parallel for simd reduction(+:dummy_sum)
        for (size_t i = 0; i < benchmark_size; ++i) { dummy_sum += h_buffer[i]; }
    } else {
        DATA_TYPE* d_bench_in;
        DATA_TYPE* d_bench_out;
        cudaStream_t bench_stream;
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
        
        cudaFree(d_bench_in);
        cudaFree(d_bench_out);
        cudaStreamDestroy(bench_stream);
    }
    
    auto end_bench = std::chrono::high_resolution_clock::now();
    worker_benchmark_times[worker_id] = std::chrono::duration<double>(end_bench - start_bench).count();
    
    #pragma omp barrier

    // PHASE 2: Load Balancing Calculation
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
            if (i == total_workers - 1) {
                worker_sizes[i] = num_elements - current_offset;
            } else {
                worker_sizes[i] = (size_t)((weights[i] / total_weight) * num_elements);
            }
            current_offset += worker_sizes[i];
        }
        if (do_trace) printf("[TRACE] EXP013_V1: Profiling complete. CPU_0 weight calculated.\n");
    }
    #pragma omp barrier

    // Allocate VRAM based on assigned proportional size
    if (!is_cpu_worker && worker_sizes[worker_id] > 0) {
        CUDA_CHECK(cudaMalloc(&d_inputs[worker_id], worker_sizes[worker_id] * sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMalloc(&d_outputs[worker_id], sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaStreamCreate(&streams[worker_id]));
    }
}

void algorithm_execute(DATA_TYPE* h_buffer, size_t num_elements, int block_size, bool dedicated_threads, 
                       int worker_id, int total_workers, bool is_cpu_worker, int gpu_id, bool do_trace, 
                       double& out_result) {
    out_result = 0.0;
    size_t my_size = worker_sizes[worker_id];
    size_t my_offset = worker_offsets[worker_id];
    if (my_size == 0) return;

    if (is_cpu_worker) {
        DATA_TYPE chunk_sum = 0;
        int nested_threads = dedicated_threads ? std::max(1, omp_get_num_procs() - (total_workers - 1)) : omp_get_num_procs();
        #pragma omp parallel for simd num_threads(nested_threads) reduction(+:chunk_sum)
        for (size_t j = 0; j < my_size; ++j) {
            chunk_sum += h_buffer[my_offset + j];
        }
        out_result = static_cast<double>(chunk_sum);
    } else {
        DATA_TYPE zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_outputs[worker_id], &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id]));
        CUDA_CHECK(cudaMemcpyAsync(d_inputs[worker_id], h_buffer + my_offset, my_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id]));

        int grid = (my_size + block_size - 1) / block_size;
        if (grid > 1024) grid = 1024;
        
        reduce_kernel<<<grid, block_size, block_size * sizeof(DATA_TYPE), streams[worker_id]>>>(
            d_inputs[worker_id], my_size, d_outputs[worker_id]
        );

        DATA_TYPE h_gpu_output = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_gpu_output, d_outputs[worker_id], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[worker_id]));
        CUDA_CHECK(cudaStreamSynchronize(streams[worker_id]));
        out_result = static_cast<double>(h_gpu_output);
    }
}

void algorithm_teardown(int worker_id, bool is_cpu_worker, int gpu_id) {
    if (!is_cpu_worker && worker_sizes[worker_id] > 0) {
        CUDA_CHECK(cudaFree(d_inputs[worker_id]));
        CUDA_CHECK(cudaFree(d_outputs[worker_id]));
        CUDA_CHECK(cudaStreamDestroy(streams[worker_id]));
    }
}
#endif