#include <cpp_header.h>
#include <cuda_header.cuh>
#include <shared_header.h>

int main()
{
    auto dim = 1000;

    auto [input, filter] = generate(dim);

    int width = dim;
    int height = dim;

    // Allocate host memory for input image, filter, and output
    const auto h_input = input;
    const auto h_filter = filter;
    const auto h_output_basic = new float[width * height];

    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel execution parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    size_t sharedMemorySize = (blockDim.x + 2 * FILTER_RADIUS) * (blockDim.y + 2 * FILTER_RADIUS) * sizeof(float);

    // Run kernels
    std::pair<int, int> inputSize = {width, height};
    std::pair<int, int> filterParams = {FILTER_SIZE, FILTER_RADIUS};

    size_t free_mem_before, total_mem;
    cudaMemGetInfo(&free_mem_before, &total_mem);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    convolution2D_basic<<<gridDim, blockDim, sharedMemorySize>>>(d_input, d_filter, d_output, inputSize, filterParams);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Memory Usage After Kernel
    size_t free_mem_after;
    cudaMemGetInfo(&free_mem_after, &total_mem);

    std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;
    std::cout << "Memory used: " << (free_mem_before - free_mem_after) / (1024 * 1024) << " MB" << std::endl;

    cudaMemcpy(h_output_basic, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    delete[] h_output_basic;
    delete[] input;
    delete[] filter;

    return 0;
}
