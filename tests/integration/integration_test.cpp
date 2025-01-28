#include <gtest/gtest.h>
#include <cuda_runtime.h>

// Assume these are implemented elsewhere
void generate(float*& input, float*& filter, int size);
void convolution(const float* input, const float* filter, float* output, int size);
void cpu_convolution(const float* input, const float* filter, float* output, int size);

// Integration test
TEST(IntegrationTest, ConvolutionMatchesCPU) {
    int size = 1024;
    float *input, *filter, *cuda_output, *cpu_output;

    // Allocate unified memory
    cudaMallocManaged(&input, size * sizeof(float));
    cudaMallocManaged(&filter, size * sizeof(float));
    cudaMallocManaged(&cuda_output, size * sizeof(float));
    cudaMallocManaged(&cpu_output, size * sizeof(float));

    // Generate input and filter
    generate(input, filter, size);

    // Run both versions
    convolution(input, filter, cuda_output, size);
    cpu_convolution(input, filter, cpu_output, size);

    // Synchronize CUDA execution
    cudaDeviceSynchronize();

    // Check results
    for (int i = 0; i < size; i++) {
        EXPECT_NEAR(cuda_output[i], cpu_output[i], 1e-5);
    }

    // Free memory
    cudaFree(input);
    cudaFree(filter);
    cudaFree(cuda_output);
    cudaFree(cpu_output);
}