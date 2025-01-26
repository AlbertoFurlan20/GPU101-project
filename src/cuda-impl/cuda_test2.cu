#include "cuda_implementation.cuh"
#include <iostream>
#include <cuda_runtime.h>

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

// Tiling parameters (adjust based on device capabilities)
#define TILE_WIDTH 32  // You can adjust this based on device capabilities

// CUDA kernel for 2D convolution
__global__ void convolution2D_kernel2(float *d_input, float *d_output, float *d_filter,
                                      int width, int height, int filterSize) {
    __shared__ float shared_input[TILE_WIDTH + FILTER_SIZE - 1][TILE_WIDTH + FILTER_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate global row and column for the output
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Load data into shared memory, with padding to handle filter overlap
    for (int i = ty; i < TILE_WIDTH + FILTER_SIZE - 1; i += blockDim.y) {
        for (int j = tx; j < TILE_WIDTH + FILTER_SIZE - 1; j += blockDim.x) {
            int input_row = row + i - FILTER_RADIUS;
            int input_col = col + j - FILTER_RADIUS;
            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                shared_input[i][j] = d_input[input_row * width + input_col];
            } else {
                shared_input[i][j] = 0.0f;  // Pad with zeros for out-of-bound accesses
            }
        }
    }

    __syncthreads();  // Synchronize threads to ensure shared memory is fully loaded

    // Perform the convolution if the current thread corresponds to a valid output pixel
    if (ty < TILE_WIDTH && tx < TILE_WIDTH && row < height && col < width) {
        float conv_result = 0.0f;
        for (int i = 0; i < filterSize; ++i) {
            for (int j = 0; j < filterSize; ++j) {
                conv_result += shared_input[ty + i][tx + j] * d_filter[i * filterSize + j];
            }
        }
        d_output[row * width + col] = conv_result;
    }
}

void convolution2D2(float *h_input, float *h_output, float *h_filter,
                   int width, int height, int filterSize) {
    float *d_input, *d_output, *d_filter;
    size_t inputSize = width * height * sizeof(float);
    size_t filterSizeInBytes = filterSize * filterSize * sizeof(float);
    size_t outputSize = width * height * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);
    cudaMalloc(&d_filter, filterSizeInBytes);

    // Copy input data and filter to device
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSizeInBytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);  // Each block processes a TILE_WIDTH x TILE_WIDTH region
    int gridX = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int gridY = (height + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 gridSize(gridX, gridY);

    // Launch the convolution kernel
    convolution2D_kernel2<<<gridSize, blockSize>>>(d_input, d_output, d_filter, width, height, filterSize);

    // Synchronize to ensure kernel execution is complete before copying data back
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}

int main_test2(int dim) {
    // Example configuration (using smaller image for testing)
    int width = dim; // Example image width (64x64 for testing, scale up as needed)
    int height = dim; // Example image height (64x64)
    int filterSize = FILTER_SIZE;

    // Allocate host memory for input image, filter, and output
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    float *h_filter = new float[filterSize * filterSize];

    // Initialize input data and filter (example)
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;  // Random data for input image
    }

    // Initialize filter (example: simple averaging filter)
    for (int i = 0; i < filterSize * filterSize; ++i) {
        h_filter[i] = 1.0f / (filterSize * filterSize);  // Average filter
    }

    // Perform 2D convolution
    convolution2D2(h_input, h_output, h_filter, width, height, filterSize);

    // Example output (just print the first few values)
    std::cout << "Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_input;
    delete[] h_output;
    delete[] h_filter;

    return 0;
}