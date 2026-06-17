#ifndef KERNEL_CUH
#define KERNEL_CUH

// Naive Shared Memory Reduction Kernel
template <typename T>
__global__ void shared_mem_sum_kernel(const T* input, T* output, size_t n) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        customAtomicAdd(output, sdata[0]);
    }
}

// Wrapper execution function
void execute_algorithm(const DATA_TYPE* h_input, size_t n, int reps, int warmup, int block_size, bool dedicated_threads, double& avg_time_us, double& final_result) {
    
    DATA_TYPE* d_input = nullptr;
    DATA_TYPE* d_output = nullptr;
    
    // Allocate VRAM. We know 'n' is safe because the main wrapper checked cudaMemGetInfo.
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(DATA_TYPE)));

    int num_blocks = (n + block_size - 1) / block_size;
    size_t smem_size = block_size * sizeof(DATA_TYPE);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DATA_TYPE h_gpu_output = 0;
    double total_time_ms = 0;

    // Measurement loop
    for (int r = 0; r < warmup + reps; r++) {
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(DATA_TYPE), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // START TIMER
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Transfer RAM -> VRAM (Measured)
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, stream));

        // 2. Kernel Execution (Measured)
        shared_mem_sum_kernel<<<num_blocks, block_size, smem_size, stream>>>(d_input, d_output, n);

        // 3. Transfer VRAM -> RAM (Measured)
        CUDA_CHECK(cudaMemcpyAsync(&h_gpu_output, d_output, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // STOP TIMER
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (r >= warmup) {
            std::chrono::duration<double> duration = end_time - start_time;
            total_time_ms += (duration.count() * 1000.0);
        }
    }

    final_result = static_cast<double>(h_gpu_output);
    avg_time_us = (total_time_ms * 1000.0) / reps;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

#endif