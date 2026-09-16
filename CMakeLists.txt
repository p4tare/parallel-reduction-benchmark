cmake_minimum_required(VERSION 3.24)
project(parallel_reduction_benchmark_v3_0 LANGUAGES CXX)

option(PRBENCH_ENABLE_CUDA "Build NVIDIA CUDA backends" ON)
option(PRBENCH_NATIVE_CPU_TUNING "Enable -march=native on GCC/Clang" ON)
option(PRBENCH_ENABLE_GDS "Build optional GPUDirect Storage worker when cuFile is available" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP QUIET)
find_package(Threads REQUIRED)

set(PRBENCH_SOURCES
    native/src/main.cpp
    native/src/cli.cpp
    native/src/dataset.cpp
    native/src/cpu_backend.cpp
    native/src/strategy.cpp
)

if(PRBENCH_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        list(APPEND PRBENCH_SOURCES native/src/gpu_backend.cu)
        set(PRBENCH_HAS_CUDA TRUE)
    else()
        message(WARNING "CUDA requested but no CUDA compiler was found; building CPU-only worker")
        list(APPEND PRBENCH_SOURCES native/src/gpu_backend_stub.cpp)
        set(PRBENCH_HAS_CUDA FALSE)
    endif()
else()
    list(APPEND PRBENCH_SOURCES native/src/gpu_backend_stub.cpp)
    set(PRBENCH_HAS_CUDA FALSE)
endif()

add_executable(prbench-worker ${PRBENCH_SOURCES})
target_include_directories(prbench-worker PRIVATE native/include)
target_compile_options(prbench-worker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)

if(PRBENCH_NATIVE_CPU_TUNING AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(prbench-worker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(prbench-worker PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_OPENMP=1)
else()
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_OPENMP=0)
endif()

if(PRBENCH_HAS_CUDA)
    set_target_properties(prbench-worker PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(prbench-worker PRIVATE CUDA::cudart)
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_CUDA=1)
    target_compile_options(prbench-worker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

    # Experimental worker used by the GPU break-even branch. It deliberately keeps
    # reduction backend fixed at CUB while changing only the data-access/memory policy.
    add_executable(prbench-memory-path-worker native/src/memory_path_worker.cu)
    set_target_properties(prbench-memory-path-worker PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(prbench-memory-path-worker PRIVATE CUDA::cudart)
    target_compile_options(prbench-memory-path-worker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

    if(PRBENCH_ENABLE_GDS)
        find_path(PRBENCH_CUFILE_INCLUDE_DIR cufile.h
            HINTS
                ${CUDAToolkit_INCLUDE_DIRS}
                /usr/local/cuda/include
        )
        find_library(PRBENCH_CUFILE_LIBRARY cufile
            HINTS
                ${CUDAToolkit_LIBRARY_DIR}
                /usr/local/cuda/lib64
                /usr/local/cuda/targets/x86_64-linux/lib
        )
        if(PRBENCH_CUFILE_INCLUDE_DIR AND PRBENCH_CUFILE_LIBRARY)
            add_executable(prbench-gds-worker native/src/gds_worker.cu)
            target_include_directories(prbench-gds-worker PRIVATE ${PRBENCH_CUFILE_INCLUDE_DIR})
            set_target_properties(prbench-gds-worker PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
            )
            target_link_libraries(prbench-gds-worker PRIVATE CUDA::cudart ${PRBENCH_CUFILE_LIBRARY})
            target_compile_options(prbench-gds-worker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
            message(STATUS "GPUDirect Storage worker enabled")
        else()
            message(STATUS "GPUDirect Storage/cuFile not found; prbench-gds-worker will not be built")
        endif()
    endif()
else()
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_CUDA=0)
endif()

target_link_libraries(prbench-worker PRIVATE Threads::Threads)

enable_testing()
add_test(NAME worker_self_test COMMAND prbench-worker --self-test)
