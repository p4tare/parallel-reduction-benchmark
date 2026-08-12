#define IS_HETEROGENEOUS_AWARE 1

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <omp.h>

#define MAX_WORKERS 32

// Global Topology Maps for Peer-to-Peer Access
static int worker_to_gpu_id[MAX_WORKERS];
static DATA_TYPE* gpu_vram_partitions[MAX_WORKERS]; 

// Standard Atomic Queues
static std::atomic<size_t> worker_offsets[MAX_WORKERS];
static size_t worker_max_limits[MAX_WORKERS]; 
static size_t worker_base_starts[MAX_WORKERS];
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

    // Relying on customAtomicAdd provided globally by the Wrapper!
    if (tid == 0) { customAtomicAdd(global_sum, sdata[0]); }
}

// HETEROGENEOUS DYNAMIC EXECUTION WITH P2P WORK STEALING
void execute_algorithm(DATA_TYPE* h_buffer, size_t num_elements, int repetitions, int warmup, 
                       int block_size, bool dedicated_threads, int worker_id, int total_workers, 
                       bool is_cpu_worker, int gpu_id, bool do_trace, double& out_time_us, double& out_result) {
    
    size_t CHUNK_SIZE = 1000000; // 1 Million elements per chunk
    if (CHUNK_SIZE > num_elements / total_workers) {
        CHUNK_SIZE = num_elements / (total_workers * 10);
        if (CHUNK_SIZE == 0) CHUNK_SIZE = 256;
    }

    double total_ms = 0.0;
    DATA_TYPE final_worker_val = 0;

    // Local buffers for results and emergency fallback transfers
    DATA_TYPE* d_result = nullptr;
    DATA_TYPE* d_cpu_steal_buffer = nullptr; 
    
    if (!is_cpu_worker) {
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMalloc(&d_cpu_steal_buffer, CHUNK_SIZE * sizeof(DATA_TYPE)));
    }

    size_t shared_mem_size = block_size * sizeof(DATA_TYPE);

    for (int iter = 0; iter < warmup + repetitions; ++iter) {
        bool is_warmup = (iter < warmup);
        bool is_trace_iter = (do_trace && iter == 0);

        // ==============================================================
        // PHASE 1: TOPOLOGY MAPPING AND P2P TUNNEL SETUP
        // ==============================================================
        worker_to_gpu_id[worker_id] = is_cpu_worker ? -1 : gpu_id;
        
        #pragma omp barrier

        if (worker_id == 0 && is_trace_iter) {
            printf("\n[TRACE] ========================================================\n");
            printf("[TRACE] PHASE 1: CROSS-SOCKET P2P TOPOLOGY INITIALIZATION\n");
            printf("[TRACE] ========================================================\n");
        }
        #pragma omp barrier

        if (!is_cpu_worker) {
            for (int i = 0; i < total_workers; ++i) {
                if (i != worker_id && worker_to_gpu_id[i] != -1) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(worker_to_gpu_id[i], 0);
                    if (is_trace_iter) {
                        if (err == cudaSuccess) {
                            printf("[TRACE] Worker %d (GPU_%d): SUCCESSFULLY opened hardware P2P tunnel to GPU_%d!\n", worker_id, gpu_id, worker_to_gpu_id[i]);
                        } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
                            printf("[TRACE] Worker %d (GPU_%d): P2P tunnel to GPU_%d was already open.\n", worker_id, gpu_id, worker_to_gpu_id[i]);
                        } else {
                            printf("[TRACE] Worker %d (GPU_%d): WARNING! No hardware P2P support to GPU_%d. Falling back to PCIe transfers!\n", worker_id, gpu_id, worker_to_gpu_id[i]);
                        }
                    }
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError(); // Clear error state for older architectures
                    }
                }
            }
        }
        #pragma omp barrier

        // ==============================================================
        // PHASE 2: DATA PRE-CACHING (STATIC INITIAL PARTITIONING)
        // ==============================================================
        if (worker_id == 0) {
            if (is_trace_iter) {
                printf("\n[TRACE] ========================================================\n");
                printf("[TRACE] PHASE 2: VRAM PRE-CACHING (STATIC PARTITIONING)\n");
                printf("[TRACE] ========================================================\n");
            }
            size_t elements_per_worker = num_elements / total_workers;
            size_t remainder = num_elements % total_workers;
            for (int i = 0; i < total_workers; ++i) {
                size_t my_start = i * elements_per_worker;
                size_t my_el = elements_per_worker + (i == total_workers - 1 ? remainder : 0);
                
                worker_base_starts[i] = my_start;
                worker_offsets[i].store(0); // Offset is now RELATIVE to the partition
                worker_max_limits[i] = my_el;
            }
            global_accumulated_result.store(0);
        }
        #pragma omp barrier

        // CPU does not allocate. GPUs pull their massive partitions into VRAM
        if (!is_cpu_worker) {
            size_t my_partition_size = worker_max_limits[worker_id];
            CUDA_CHECK(cudaMalloc(&gpu_vram_partitions[worker_id], my_partition_size * sizeof(DATA_TYPE)));
            CUDA_CHECK(cudaMemcpy(gpu_vram_partitions[worker_id], h_buffer + worker_base_starts[worker_id], 
                                  my_partition_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
            
            if (is_trace_iter) {
                double size_mb = (my_partition_size * sizeof(DATA_TYPE)) / (1024.0 * 1024.0);
                printf("[TRACE] Worker %d (GPU_%d): Pre-caching complete. Loaded %.2f MB directly into VRAM.\n", worker_id, gpu_id, size_mb);
            }
        } else if (is_trace_iter) {
            double size_mb = (worker_max_limits[worker_id] * sizeof(DATA_TYPE)) / (1024.0 * 1024.0);
            printf("[TRACE] Worker %d (CPU_MASTER): Reserved a %.2f MB partition in RAM for local processing.\n", worker_id, size_mb);
        }
        #pragma omp barrier 

        if (worker_id == 0 && is_trace_iter) {
            printf("\n[TRACE] ========================================================\n");
            printf("[TRACE] PHASE 3: DYNAMIC EXECUTION AND P2P WORK STEALING\n");
            printf("[TRACE] ========================================================\n");
        }
        #pragma omp barrier

        // ==============================================================
        // PHASE 3: DYNAMIC EXECUTION
        // ==============================================================
        DATA_TYPE local_iteration_sum = 0;
        int chunks_own = 0, chunks_stolen = 0;
        bool out_of_local_work = false;
        
        int target_queue = worker_id;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            size_t current_relative_offset = worker_offsets[target_queue].fetch_add(CHUNK_SIZE);
            size_t my_elements = 0;
            
            if (current_relative_offset < worker_max_limits[target_queue]) {
                my_elements = std::min(CHUNK_SIZE, worker_max_limits[target_queue] - current_relative_offset);
                
                if (target_queue == worker_id) {
                    chunks_own++;
                } else {
                    chunks_stolen++;
                }

                // -------------------------------------------------------------
                // CPU EXECUTION NODE
                // -------------------------------------------------------------
                if (is_cpu_worker) {
                    DATA_TYPE chunk_sum = 0;
                    size_t global_ram_offset = worker_base_starts[target_queue] + current_relative_offset;
                    
                    if (target_queue == worker_id || worker_to_gpu_id[target_queue] == -1) {
                        // Step 1: Work on local RAM (or steal from another CPU socket in the future)
                        int nested_threads = dedicated_threads ? std::max(1, omp_get_num_procs() - (total_workers - 1)) : omp_get_num_procs();
                        #pragma omp parallel for simd num_threads(nested_threads) reduction(+:chunk_sum)
                        for (size_t j = 0; j < my_elements; ++j) { chunk_sum += h_buffer[global_ram_offset + j]; }
                        
                        if (is_trace_iter && chunks_own == 1) {
                            printf("[TRACE] Worker %d (CPU_MASTER): Started processing local queue using %d nested threads.\n", worker_id, nested_threads);
                        }
                    } else {
                        // Step 2: CPU steals a chunk from a GPU (Emergency D2H PCIe Fallback)
                        DATA_TYPE* h_temp = new DATA_TYPE[my_elements];
                        CUDA_CHECK(cudaMemcpy(h_temp, gpu_vram_partitions[target_queue] + current_relative_offset, my_elements * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
                        
                        int nested_threads = dedicated_threads ? std::max(1, omp_get_num_procs() - (total_workers - 1)) : omp_get_num_procs();
                        #pragma omp parallel for simd num_threads(nested_threads) reduction(+:chunk_sum)
                        for (size_t j = 0; j < my_elements; ++j) { chunk_sum += h_temp[j]; }
                        delete[] h_temp;
                        
                        if (is_trace_iter && chunks_stolen <= 100) {
                            double chunk_mb = (my_elements * sizeof(DATA_TYPE)) / (1024.0 * 1024.0);
                            printf("[TRACE] Worker %d (CPU_MASTER): EMERGENCY STEAL (D2H)! Fetched %.2f MB from Worker %d (GPU_%d) via PCIe.\n", worker_id, chunk_mb, target_queue, worker_to_gpu_id[target_queue]);
                        }
                    }
                    local_iteration_sum += chunk_sum;
                } 
                // -------------------------------------------------------------
                // GPU EXECUTION NODE (The GPUDirect Magic)
                // -------------------------------------------------------------
                else {
                    DATA_TYPE zero = 0;
                    CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
                    int grid = (my_elements + block_size - 1) / block_size;
                    if (grid > 1024) grid = 1024;
                    
                    if (worker_to_gpu_id[target_queue] != -1) {
                        // CASE 1: True P2P (GPU processes from its own VRAM, or from another GPU's VRAM bypassing CPU)
                        const DATA_TYPE* d_target_ptr = gpu_vram_partitions[target_queue] + current_relative_offset;
                        reduce_kernel<<<grid, block_size, shared_mem_size>>>(d_target_ptr, my_elements, d_result);
                        
                        if (is_trace_iter && target_queue != worker_id && chunks_stolen <= 3) {
                            double chunk_mb = (my_elements * sizeof(DATA_TYPE)) / (1024.0 * 1024.0);
                            printf("[TRACE] Worker %d (GPU_%d): HARDWARE P2P STEAL (Zero-Copy)! Direct access to %.2f MB from Worker %d (GPU_%d).\n", 
                                   worker_id, gpu_id, chunk_mb, target_queue, worker_to_gpu_id[target_queue]);
                        }
                    } else {
                        // CASE 2: GPU steals a task from the CPU queue (Standard H2D)
                        size_t global_ram_offset = worker_base_starts[target_queue] + current_relative_offset;
                        CUDA_CHECK(cudaMemcpy(d_cpu_steal_buffer, h_buffer + global_ram_offset, my_elements * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
                        reduce_kernel<<<grid, block_size, shared_mem_size>>>(d_cpu_steal_buffer, my_elements, d_result);
                        
                        if (is_trace_iter && chunks_stolen <= 3) {
                            double chunk_mb = (my_elements * sizeof(DATA_TYPE)) / (1024.0 * 1024.0);
                            printf("[TRACE] Worker %d (GPU_%d): H2D STEAL! Transferred %.2f MB from Worker %d (CPU_MASTER) via PCIe.\n", worker_id, gpu_id, chunk_mb, target_queue);
                        }
                    }
                    
                    CUDA_CHECK(cudaDeviceSynchronize());
                    DATA_TYPE chunk_sum = 0;
                    CUDA_CHECK(cudaMemcpy(&chunk_sum, d_result, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
                    local_iteration_sum += chunk_sum;
                }
            } else {
                // If this is the first time the local queue is depleted - report it in the trace
                if (target_queue == worker_id && !out_of_local_work && is_trace_iter) {
                    out_of_local_work = true;
                    const char* role = is_cpu_worker ? "CPU_MASTER" : "GPU";
                    printf("[TRACE] Worker %d (%s) DEPLETED local queue. Initiating topology search...\n", worker_id, role);
                }

                // QUEUE EMPTY -> STEAL FROM OTHERS
                bool found = false;
                for (int i = 1; i < total_workers; ++i) {
                    int candidate = (worker_id + i) % total_workers;
                    if (worker_offsets[candidate].load() < worker_max_limits[candidate]) {
                        target_queue = candidate; 
                        found = true;
                        break;
                    }
                }
                if (!found) break; // Entire cluster processed!
            }
        }

        auto stop_time = std::chrono::high_resolution_clock::now();

        // End of iteration accumulation
        #pragma omp critical
        {
            double current_global = global_accumulated_result.load();
            global_accumulated_result.store(current_global + (double)local_iteration_sum);
        }
        
        #pragma omp barrier
        
        if (worker_id == 0 && is_trace_iter) {
            printf("\n[TRACE] ========================================================\n");
            printf("[TRACE] PHASE 4: ITERATION DYNAMICS SUMMARY\n");
            printf("[TRACE] ========================================================\n");
        }
        #pragma omp barrier

        if (is_trace_iter) {
            const char* role = is_cpu_worker ? "CPU_MASTER" : "GPU";
            int gpu_str = is_cpu_worker ? -1 : gpu_id;
            printf("[TRACE] REPORT | Worker: %d (%s_%d) | Own chunks processed: %d | STOLEN CHUNKS: %d\n", worker_id, role, gpu_str, chunks_own, chunks_stolen);
        }

        // Clean up the massive VRAM caches before the next warmup/iteration
        #pragma omp barrier
        if (!is_cpu_worker) {
            CUDA_CHECK(cudaFree(gpu_vram_partitions[worker_id]));
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
        CUDA_CHECK(cudaFree(d_result));
        CUDA_CHECK(cudaFree(d_cpu_steal_buffer));
    }

    out_time_us = (total_ms * 1000.0) / repetitions;
    out_result = static_cast<double>(final_worker_val);
}