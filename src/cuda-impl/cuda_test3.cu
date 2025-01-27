#include "cuda_implementation.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];

__global__ void tiledConvolution2D(const float* d_input, float* d_output, int width, int height) {
    // Shared memory for the tile
    extern __shared__ float sharedMem[];

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    int sharedWidth = blockDim.x + 2 * FILTER_RADIUS;

    // Load data into shared memory
    int sharedRow = ty + FILTER_RADIUS;
    int sharedCol = tx + FILTER_RADIUS;

    // Global indices with boundary check
    int inputRow = min(max(row, 0), height - 1);
    int inputCol = min(max(col, 0), width - 1);

    sharedMem[sharedRow * sharedWidth + sharedCol] = d_input[inputRow * width + inputCol];

    // Load halo cells
    if (tx < FILTER_RADIUS) {
        int leftCol = max(col - FILTER_RADIUS, 0);
        sharedMem[sharedRow * sharedWidth + tx] = d_input[inputRow * width + leftCol];
        int rightCol = min(col + blockDim.x, width - 1);
        sharedMem[sharedRow * sharedWidth + sharedCol + blockDim.x] = d_input[inputRow * width + rightCol];
    }

    if (ty < FILTER_RADIUS) {
        int topRow = max(row - FILTER_RADIUS, 0);
        int bottomRow = min(row + blockDim.y, height - 1);
        sharedMem[ty * sharedWidth + sharedCol] = d_input[topRow * width + inputCol];
        sharedMem[(sharedRow + blockDim.y) * sharedWidth + sharedCol] = d_input[bottomRow * width + inputCol];
    }

    __syncthreads();

    // Perform convolution
    if (row < height && col < width) {
        float result = 0.0f;
        for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++) {
            for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++) {
                int sharedY = sharedRow + fy;
                int sharedX = sharedCol + fx;
                int filterIdx = (fy + FILTER_RADIUS) * FILTER_SIZE + (fx + FILTER_RADIUS);
                result += sharedMem[sharedY * sharedWidth + sharedX] * d_filter[filterIdx];
            }
        }
        d_output[row * width + col] = result;
    }
}

void convolution2D(const float* h_input, const float* h_filter, float* h_output, int width, int height) {
    // Allocate device memory
    float *d_input, *d_output;
    size_t inputSize = width * height * sizeof(float);
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, inputSize);

    // Copy input and filter to device
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Configure kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    size_t sharedMemSize = (blockDim.x + 2 * FILTER_RADIUS) * (blockDim.y + 2 * FILTER_RADIUS) * sizeof(float);

    // Launch kernel with streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    tiledConvolution2D<<<gridDim, blockDim, sharedMemSize, stream>>>(d_input, d_output, width, height);

    // Synchronize and copy result back to host
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_output, d_output, inputSize, cudaMemcpyDeviceToHost);

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main_test3(int dim, float* input, float* filter) {
    // Example usage
    int width = dim;
    int height = dim;
    const int size = width * height;
    float *h_input = input;
    float *h_output = new float[width * height];
    // float *h_filter = new float[filterSize * filterSize];
    float *h_filter = filter;

    convolution2D(h_input, h_filter, h_output, width, height);

    // Example output (just print the first few values)
    std::cout << "Output (first 10 values):" << std::endl;
    std::cout << "> (input) [ ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << " ]\n";
    std::cout << "> (filter) [ ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_filter[i] << " ";
    }
    std::cout << " ]\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // delete[] h_input;
    // delete[] h_filter;
    delete[] h_output;
    return 0;
}
