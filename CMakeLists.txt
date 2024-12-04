cmake_minimum_required(VERSION 3.22.1)

# Set CUDA compiler before the project declaration
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(GPU101_project CUDA CXX)

# Enable CUDA language
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(GPU101_project main.cu)

set_target_properties(GPU101_project PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
