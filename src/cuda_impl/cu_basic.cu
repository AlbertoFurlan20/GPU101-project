#include <cuda_header.cuh>

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

float* main_basic(const int dim, const float* input, const float* filter)
{
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
    convolution2D_basic<<<gridDim, blockDim, sharedMemorySize>>>(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS});
    cudaMemcpy(h_output_basic, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return h_output_basic;
}
