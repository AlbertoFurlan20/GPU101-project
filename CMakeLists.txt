cmake_minimum_required(VERSION 3.22.1)

# Set CUDA compiler before the project declaration
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(GPU101_project CUDA CXX)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setting sources files
set(SOURCES_CPP
        src/cpp-impl/cpp_implementation.h
        src/cpp-impl/cpp_implementation.cpp
)

set(SOURCES_CUDA
        src/cuda-impl/cuda_implementation.cuh
        src/cuda-impl/cuda_implementation.cu
)

add_executable(
        GPU101_project-CUDA
        main.cu
        ${SOURCES_CUDA}
)

set_target_properties(GPU101_project-CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(
        GPU101_project-CPP
        main.cpp
        ${SOURCES_CPP}
)

