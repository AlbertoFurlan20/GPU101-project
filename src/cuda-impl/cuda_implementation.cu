#include <iostream>
#include <cstdlib>
#include <cassert>

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

__global__
void convolution_cuda(const input_type *input, const input_type *filter, input_type *output, const int width, const int height, const int filter_size, const int filter_radius)
{
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width)
    {
        input_type value = 0.0f;
        for (int row = 0; row < filter_size; row++)
        {
            for (int col = 0; col < filter_size; col++)
            {
                int inRow = outRow - filter_radius + row;
                int inCol = outCol - filter_radius + col;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                {
                    value += filter[row * filter_size + col] * input[inRow * width + inCol];
                }
            }
        }
        output[outRow * width + outCol] = value;
    }
}

void a_checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int run_cuda_main()
{
    const unsigned dim = 2;
    const unsigned int width = dim;
    const unsigned int height = dim;

    input_type *input = new input_type[width * height];               // Input
    filter_type *filter = new filter_type[FILTER_SIZE * FILTER_SIZE]; // Convolution filter
    input_type *output_gpu = new input_type[width * height];          // Output (GPU)

    // Randomly initialize the inputs
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
        filter[i] = static_cast<filter_type>(rand()) / RAND_MAX;

    for (int i = 0; i < width * height; ++i)
        input[i] = static_cast<input_type>(rand()) / RAND_MAX; // Random value between 0 and 1

    // Allocate device memory
    input_type *d_input, *d_filter, *d_output;
    a_checkCudaErrors(cudaMalloc(&d_input, width * height * sizeof(input_type)));
    a_checkCudaErrors(cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(filter_type)));
    a_checkCudaErrors(cudaMalloc(&d_output, width * height * sizeof(input_type)));

    // Copy data to device
    a_checkCudaErrors(cudaMemcpy(d_input, input, width * height * sizeof(input_type), cudaMemcpyHostToDevice));
    a_checkCudaErrors(cudaMemcpy(d_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(filter_type), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); // Calculate grid size

    convolution_cuda<<<gridDim, blockDim>>>(d_input, d_filter, d_output, width, height, FILTER_SIZE, FILTER_RADIUS);
    a_checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back to host
    a_checkCudaErrors(cudaMemcpy(output_gpu, d_output, width * height * sizeof(input_type), cudaMemcpyDeviceToHost));

    // Cleanup
    delete[] input;
    delete[] filter;
    delete[] output_gpu;

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}