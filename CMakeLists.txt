cmake_minimum_required(VERSION 3.22.1)

# Set the project
project(GPU101_project CUDA CXX)

# Set CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Platform-specific CUDA setup
if (APPLE)
    message(STATUS "[CONFIG] running configuration for APPLE device")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
elseif (UNIX)
    message(STATUS "[CONFIG] running configuration for UNIX device")
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
elseif (WIN32)
    message(STATUS "[CONFIG] running configuration for WINDOWS device")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
else ()
    message(STATUS "[CONFIG] No default config needed")
endif ()

# Find and configure CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# Find OpenMP and enable it if found
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "[CONFIG] OpenMP found")
else()
    message(STATUS "[CONFIG] OpenMP not found")
endif()

# Setting sources files
set(SOURCES_CUDA
        src/cuda-impl/cuda_header.cuh
        src/cpp-impl/cpp_header.h
        src/cuda-impl/cu_shared_version.cu
        src/cuda-impl/cu_tiling_version.cu
        src/cuda-impl/cu_stream_tiling_version.cu
        src/cuda-impl/cu_basic.cu
        src/cuda-impl/cu_streams_version.cu
        src/shared/status-check.cpp
        src/shared/generator.cpp
        src/cpp-impl/c_version.cpp
)

# Add the executable
add_executable(
        GPU101_project-CUDA
        main.cpp
        ${SOURCES_CUDA}
)

# Enable separable compilation for CUDA
set_target_properties(GPU101_project-CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(GPU101_project-CUDA PUBLIC OpenMP::OpenMP_CXX)
endif()