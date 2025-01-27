#include "cuda_implementation.cuh"
#include <iostream>
#include <cuda_runtime.h>

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

// Tiling parameters (adjust based on device capabilities)
#define TILE_WIDTH 32  // You can adjust this based on device capabilities

// Utility macro for 2D indexing
#define IDX_2D(x, y, width) ((y) * (width) + (x))

__global__ void convolution2D_basic(const float* input, const float* kernel, float* output,
                              std::pair<int, int> inputSize, std::pair<int, int> filterParams)
{
    extern __shared__ float sharedIndexes[];

    int kernelRadius = filterParams.second;
    int kernelSize = filterParams.first;
    int width = inputSize.first;
    int height = inputSize.second;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + kernelRadius;
    int sharedY = threadIdx.y + kernelRadius;

    int inputIndex = IDX_2D(x, y, width);

    // Load data into shared memory
    if (x < width && y < height)
    {
        sharedIndexes[sharedY * (blockDim.x + 2 * kernelRadius) + sharedX] = input[inputIndex];
    }
    else
    {
        sharedIndexes[sharedY * (blockDim.x + 2 * kernelRadius) + sharedX] = 0.0f;
    }

    // Load halo regions
    if (threadIdx.x < kernelRadius)
    {
        sharedIndexes[sharedY * (blockDim.x + 2 * kernelRadius) + (sharedX - kernelRadius)] =
            (x >= kernelRadius) ? input[IDX_2D(x - kernelRadius, y, width)] : 0.0f;
        sharedIndexes[sharedY * (blockDim.x + 2 * kernelRadius) + (sharedX + blockDim.x)] =
            (x + blockDim.x < width) ? input[IDX_2D(x + blockDim.x, y, width)] : 0.0f;
    }

    if (threadIdx.y < kernelRadius)
    {
        sharedIndexes[(sharedY - kernelRadius) * (blockDim.x + 2 * kernelRadius) + sharedX] =
            (y >= kernelRadius) ? input[IDX_2D(x, y - kernelRadius, width)] : 0.0f;
        sharedIndexes[(sharedY + blockDim.y) * (blockDim.x + 2 * kernelRadius) + sharedX] =
            (y + blockDim.y < height) ? input[IDX_2D(x, y + blockDim.y, width)] : 0.0f;
    }

    __syncthreads();

    // Apply convolution
    if (x < width && y < height)
    {
        float partialResult = 0.0f;

        for (int idxY = 0; idxY < kernelSize; ++idxY)
        {
            for (int idxX = 0; idxX < kernelSize; ++idxX)
            {
                int sharedInputY = sharedY - kernelRadius + idxY;
                int sharedInputX = sharedX - kernelRadius + idxX;
                partialResult += kernel[idxY * kernelSize + idxX] *
                    sharedIndexes[sharedInputY * (blockDim.x + 2 * kernelRadius) + sharedInputX];
            }
        }

        output[IDX_2D(x, y, width)] = partialResult;
    }
}

// Version 1: Tiling Only
__global__ void convolution2D_tiling(const float* input, const float* kernel, float* output,
                                     std::pair<int, int> inputSize, std::pair<int, int> filterParams)
{
    extern __shared__ float sharedIndexes[];

    int kernelRadius = filterParams.second;
    int kernelSize = filterParams.first;
    int width = inputSize.first;
    int height = inputSize.second;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + kernelRadius;
    int sharedY = threadIdx.y + kernelRadius;

    int sharedWidth = blockDim.x + 2 * kernelRadius;
    int sharedHeight = blockDim.y + 2 * kernelRadius;

    // Load data into shared memory, including halo regions
    for (int tileY = threadIdx.y; tileY < sharedHeight; tileY += blockDim.y)
    {
        for (int tileX = threadIdx.x; tileX < sharedWidth; tileX += blockDim.x)
        {
            int globalX = blockIdx.x * blockDim.x + tileX - kernelRadius;
            int globalY = blockIdx.y * blockDim.y + tileY - kernelRadius;

            sharedIndexes[tileY * sharedWidth + tileX] =
                (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height)
                    ? input[IDX_2D(globalX, globalY, width)]
                    : 0.0f;
        }
    }

    __syncthreads();

    // Apply convolution
    if (x < width && y < height)
    {
        float partialResult = 0.0f;

        for (int idxY = 0; idxY < kernelSize; ++idxY)
        {
            for (int idxX = 0; idxX < kernelSize; ++idxX)
            {
                int sharedInputY = sharedY - kernelRadius + idxY;
                int sharedInputX = sharedX - kernelRadius + idxX;
                partialResult += kernel[idxY * kernelSize + idxX] *
                    sharedIndexes[sharedInputY * sharedWidth + sharedInputX];
            }
        }

        output[IDX_2D(x, y, width)] = partialResult;
    }
}

// Version 2: Streams Only
void launch_convolution2D_streams(const float* input, const float* kernel, float* output,
                                  std::pair<int, int> inputSize, std::pair<int, int> filterParams,
                                  int numStreams)
{
    int width = inputSize.first;
    int height = inputSize.second;

    int streamHeight = (height + numStreams - 1) / numStreams;
    cudaStream_t* streams = new cudaStream_t[numStreams];

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
        int startRow = i * streamHeight;
        int endRow = min((i + 1) * streamHeight, height);

        if (startRow >= height) break;

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (endRow - startRow + blockDim.y - 1) / blockDim.y);

        convolution2D_basic<<<gridDim, blockDim, (blockDim.x + 2 * filterParams.second) * (blockDim.y + 2 * filterParams.
            second) * sizeof(float), streams[i]>>>(
            input + startRow * width, kernel, output + startRow * width,
            {width, endRow - startRow}, filterParams);
    }

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    delete[] streams;
}

// Version 3: Tiling + Streams
void launch_convolution2D_tiling_streams(const float* input, const float* kernel, float* output,
                                         std::pair<int, int> inputSize, std::pair<int, int> filterParams,
                                         int numStreams)
{
    int width = inputSize.first;
    int height = inputSize.second;

    int streamHeight = (height + numStreams - 1) / numStreams;
    cudaStream_t* streams = new cudaStream_t[numStreams];

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
        int startRow = i * streamHeight;
        int endRow = min((i + 1) * streamHeight, height);

        if (startRow >= height) break;

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (endRow - startRow + blockDim.y - 1) / blockDim.y);

        convolution2D_tiling<<<gridDim, blockDim, (blockDim.x + 2 * filterParams.second) * (blockDim.y + 2 *
            filterParams.second) * sizeof(float), streams[i]>>>(
            input + startRow * width, kernel, output + startRow * width,
            {width, endRow - startRow}, filterParams);
    }

    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    delete[] streams;
}

int main_test2(int dim, float* input, float* filter)
{
    // Example configuration
    int width = dim;
    int height = dim;
    int filterSize = FILTER_SIZE;

    // Allocate host memory for input image, filter, and output
    float* h_input = input;
    float* h_filter = filter;
    float* h_output_basic = new float[width * height];
    float* h_output_tiling = new float[width * height];
    float* h_output_streams = new float[width * height];
    float* h_output_both = new float[width * height];

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
    convolution2D_basic<<<gridDim, blockDim, sharedMemorySize>>>(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS});
    cudaMemcpy(h_output_basic, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    convolution2D_tiling<<<gridDim, blockDim, sharedMemorySize>>>(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS});
    cudaMemcpy(h_output_tiling, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    launch_convolution2D_streams(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS}, 4);
    cudaMemcpy(h_output_streams, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    launch_convolution2D_tiling_streams(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS}, 4);
    cudaMemcpy(h_output_both, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Example output (just print the first few values)
    std::cout << "Output (first 10 values):" << std::endl;
    std::cout << "> (input) [ ";
    for (int i = 0; i < 10; ++i) std::cout << h_input[i] << " ";
    std::cout << "]\n";
    std::cout << "> (filter) [ ";
    for (int i = 0; i < 10; ++i) std::cout << h_filter[i] << " ";
    std::cout << "]\n";
    std::cout << "> (basic) [ ";
    for (int i = 0; i < 10; ++i) std::cout << h_output_basic[i] << " ";
    std::cout << "]\n";
    std::cout << "> (out-tiling) [ ";
    for (int i = 0; i < 10; ++i) std::cout << h_output_tiling[i] << " ";
    std::cout << "]\n";
    std::cout << "> (out-streams) [ ";
    for (int i = 0; i < 10; ++i) std::cout << h_output_streams[i] << " ";
    std::cout << "]\n";
    std::cout << "> (out-both) [ ";
    for (int i = 0; i < 10; ++i) std::cout << h_output_both[i] << " ";
    std::cout << "]\n";
    std::cout << std::endl;

    // Clean up
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    delete[] h_output_basic;
    delete[] h_output_tiling;
    delete[] h_output_streams;
    delete[] h_output_both;

    return 0;
}
