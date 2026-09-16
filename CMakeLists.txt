cmake_minimum_required(VERSION 3.24)
project(parallel_reduction_benchmark_v4_0 LANGUAGES CXX)

option(PRBENCH_ENABLE_CUDA "Build NVIDIA CUDA backends" ON)
option(PRBENCH_NATIVE_CPU_TUNING "Enable -march=native on GCC/Clang" ON)
option(PRBENCH_ENABLE_GDS "Enable optional GPUDirect Storage integration when cuFile is available" ON)

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
        if(DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            get_filename_component(PRBENCH_CXX_COMPILER_REAL "${CMAKE_CXX_COMPILER}" REALPATH)
            get_filename_component(PRBENCH_CUDA_HOST_COMPILER_REAL "${CMAKE_CUDA_HOST_COMPILER}" REALPATH)
            if(NOT PRBENCH_CXX_COMPILER_REAL STREQUAL PRBENCH_CUDA_HOST_COMPILER_REAL)
                message(FATAL_ERROR
                    "prbench requires CMAKE_CXX_COMPILER and CMAKE_CUDA_HOST_COMPILER to be the same compiler. "
                    "Detected C++=${CMAKE_CXX_COMPILER}, CUDA host=${CMAKE_CUDA_HOST_COMPILER}. "
                    "On apl13 configure with: "
                    "-DCMAKE_CXX_COMPILER=/usr/bin/g++-13 "
                    "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13")
            endif()
        else()
            set(
                CMAKE_CUDA_HOST_COMPILER
                "${CMAKE_CXX_COMPILER}"
                CACHE FILEPATH
                "Host compiler used by NVCC; kept equal to CMAKE_CXX_COMPILER"
                FORCE
            )
            message(STATUS "NVCC host compiler pinned to C++ compiler: ${CMAKE_CXX_COMPILER}")
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        list(APPEND PRBENCH_SOURCES
            native/src/gpu_backend.cu
            native/src/integrated_strategy.cu
        )
        set(PRBENCH_HAS_CUDA TRUE)
    else()
        message(WARNING "CUDA requested but no CUDA compiler was found; building CPU-only worker")
        list(APPEND PRBENCH_SOURCES
            native/src/gpu_backend_stub.cpp
            native/src/integrated_strategy_stub.cpp
        )
        set(PRBENCH_HAS_CUDA FALSE)
    endif()
else()
    list(APPEND PRBENCH_SOURCES
        native/src/gpu_backend_stub.cpp
        native/src/integrated_strategy_stub.cpp
    )
    set(PRBENCH_HAS_CUDA FALSE)
endif()

add_executable(prbench-worker ${PRBENCH_SOURCES})
target_include_directories(prbench-worker PRIVATE native/include)
target_compile_options(prbench-worker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(prbench-worker PRIVATE Threads::Threads)

if(PRBENCH_NATIVE_CPU_TUNING AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(prbench-worker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(prbench-worker PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_OPENMP=1)
else()
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_OPENMP=0)
endif()

# NUMA-local registered staging belongs to the main worker in v4, not to a side worker.
find_path(PRBENCH_NUMA_INCLUDE_DIR numa.h)
find_library(PRBENCH_NUMA_LIBRARY numa)
if(PRBENCH_NUMA_INCLUDE_DIR AND PRBENCH_NUMA_LIBRARY)
    target_include_directories(prbench-worker PRIVATE ${PRBENCH_NUMA_INCLUDE_DIR})
    target_link_libraries(prbench-worker PRIVATE ${PRBENCH_NUMA_LIBRARY})
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_LIBNUMA=1)
    message(STATUS "NUMA-aware registered staging enabled in prbench-worker")
else()
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_LIBNUMA=0)
    message(STATUS "libnuma development files not found; registered staging will use portable pinned memory")
endif()

if(PRBENCH_HAS_CUDA)
    set_target_properties(prbench-worker PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(prbench-worker PRIVATE CUDA::cudart)
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_CUDA=1)
    target_compile_options(prbench-worker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)

    # Detect GDS/cuFile once for the unified worker. The runtime strategy still capability-gates
    # GDS, so a machine without cuFile remains fully usable for every other algorithm.
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
            target_include_directories(prbench-worker PRIVATE ${PRBENCH_CUFILE_INCLUDE_DIR})
            target_link_libraries(prbench-worker PRIVATE ${PRBENCH_CUFILE_LIBRARY})
            target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_GDS=1)
            message(STATUS "GPUDirect Storage/cuFile detected for unified prbench-worker")
        else()
            target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_GDS=0)
            message(STATUS "GPUDirect Storage/cuFile not found; GDS algorithms will be capability-skipped")
        endif()
    else()
        target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_GDS=0)
    endif()
else()
    target_compile_definitions(prbench-worker PRIVATE PRBENCH_HAS_CUDA=0 PRBENCH_HAS_GDS=0)
endif()

enable_testing()
add_test(NAME worker_self_test COMMAND prbench-worker --self-test)
