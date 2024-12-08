cmake_minimum_required(VERSION 3.22.1)

project(GPU101_project CUDA CXX)

# Set CUDA compiler before the project declaration
if (APPLE)
    message(STATUS "[CONFIG] running configuration for APPLE device")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

elseif (WIN32)
    message(STATUS "[CONFIG] running configuration for WINDOWS device")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

    # Find CUDA package
    message(STATUS "[CONFIG] (1.) finding CUDA")
    find_package(CUDA REQUIRED)


    # Set the include and library directories
    message(STATUS "[CONFIG] (2.) including directories")
    include_directories(${CUDA_INCLUDE_DIRS})

    message(STATUS "[CONFIG] (3.) linking directories")
    link_directories(${CUDA_LIBRARIES})

endif ()

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
        src/cuda-impl/cuda-status-check.cpp
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

