#ifndef KERNEL_CUH
#define KERNEL_CUH

#define MAX_WORKERS 32
static DATA_TYPE* d_inputs[MAX_WORKERS];
static DATA_TYPE* d_outputs[MAX_WORKERS];
static cudaStream_t streams[MAX_WORKERS];

// Naiwny Shared Memory Reduction Kernel
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
        atomicAdd(output, sdata[0]); // U¿ywa wersji wspieranej przez Wrapper
    }
}

// ----------------------------------------------------------------------------------
// 1. FAZA SETUP - Alokacja i inicjalizacja na starcie
// ----------------------------------------------------------------------------------
void algorithm_setup(DATA_TYPE* h_buffer, size_t num_elements, int block_size, bool dedicated_threads, 
                     int worker_id, int total_workers, bool is_cpu_worker, int gpu_id, bool do_trace) {
    
    // EXP002 to najprostszy stary algorytm - dzia³a tylko na pierwszej karcie GPU (Worker 1)
    if (!is_cpu_worker && worker_id == 1) {
        CUDA_CHECK(cudaMalloc(&d_inputs[worker_id], num_elements * sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMalloc(&d_outputs[worker_id], sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaStreamCreate(&streams[worker_id]));
    }
}

// ----------------------------------------------------------------------------------
// 2. FAZA OBLICZEÑ (Wywo³ywana przez Wrapper w pêtli. Wrapper zajmuje siê mierzeniem czasu)
// ----------------------------------------------------------------------------------
void algorithm_execute(DATA_TYPE* h_buffer, size_t num_elements, int block_size, bool dedicated_threads, 
                       int worker_id, int total_workers, bool is_cpu_worker, int gpu_id, bool do_trace, 
                       double& out_result) {
    
    out_result = 0.0;
    
    // Ignorujemy procesor i inne karty - algorytm 002 u¿ywa tylko jednego GPU
    if (is_cpu_worker || worker_id != 1) return;

    DATA_TYPE zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_outputs[worker_id], &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id]));
    CUDA_CHECK(cudaMemcpyAsync(d_inputs[worker_id], h_buffer, num_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice, streams[worker_id]));

    int num_blocks = (num_elements + block_size - 1) / block_size;
    size_t smem_size = block_size * sizeof(DATA_TYPE);

    shared_mem_sum_kernel<<<num_blocks, block_size, smem_size, streams[worker_id]>>>(
        d_inputs[worker_id], d_outputs[worker_id], num_elements
    );

    DATA_TYPE h_gpu_output = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_gpu_output, d_outputs[worker_id], sizeof(DATA_TYPE), cudaMemcpyDeviceToHost, streams[worker_id]));
    CUDA_CHECK(cudaStreamSynchronize(streams[worker_id]));

    out_result = static_cast<double>(h_gpu_output);
}

// ----------------------------------------------------------------------------------
// 3. FAZA TEARDOWN - Sprz¹tanie pamiêci na koñcu
// ----------------------------------------------------------------------------------
void algorithm_teardown(int worker_id, bool is_cpu_worker, int gpu_id) {
    if (!is_cpu_worker && worker_id == 1) {
        CUDA_CHECK(cudaFree(d_inputs[worker_id]));
        CUDA_CHECK(cudaFree(d_outputs[worker_id]));
        CUDA_CHECK(cudaStreamDestroy(streams[worker_id]));
    }
}

#endif