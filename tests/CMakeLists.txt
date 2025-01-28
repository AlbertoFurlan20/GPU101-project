# Define the source files for your project
set(SOURCES_CUDA
        ${CMAKE_SOURCE_DIR}/src/cuda-impl/cu_shared_version.cu
        ${CMAKE_SOURCE_DIR}/src/cuda-impl/cu_tiling_version.cu
        ${CMAKE_SOURCE_DIR}/src/cuda-impl/cu_stream_tiling_version.cu
        ${CMAKE_SOURCE_DIR}/src/cuda-impl/cu_basic.cu
        ${CMAKE_SOURCE_DIR}/src/cuda-impl/cu_streams_version.cu
)

set(SOURCES_CPP
        ${CMAKE_SOURCE_DIR}/src/cpp-impl/c_version.cpp
)

set(SHARED
        ${CMAKE_SOURCE_DIR}/src/shared/status-check.cpp
        ${CMAKE_SOURCE_DIR}/src/shared/generator.cpp
        ${CMAKE_SOURCE_DIR}/src/shared/logger.cpp
)

set(HEADERS
        ${CMAKE_SOURCE_DIR}/include/cuda_header.cuh
        ${CMAKE_SOURCE_DIR}/include/cpp_header.h
        ${CMAKE_SOURCE_DIR}/include/shared_header.h
)

# Integration test
add_executable(integration_test
        integration/integration_test.cpp
        ${SOURCES_CUDA}
        ${SOURCES_CPP}
        ${SHARED}
)

# Add the include directories
target_include_directories(integration_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link required libraries
target_link_libraries(integration_test PRIVATE gtest gtest_main OpenMP::OpenMP_CXX)

# Performance test executable (from .cu source)
add_executable(performance_test
        ${CMAKE_SOURCE_DIR}/tests/performance/performance_test.cu  # Corrected the path
        ${SOURCES_CUDA}
        ${SOURCES_CPP}
        ${SHARED}
)

# Add the include directories
target_include_directories(performance_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link required libraries
target_link_libraries(performance_test PRIVATE gtest gtest_main OpenMP::OpenMP_CXX)

# Register the tests
add_test(NAME IntegrationTest COMMAND integration_test)
add_test(NAME PerformanceTest COMMAND performance_test)