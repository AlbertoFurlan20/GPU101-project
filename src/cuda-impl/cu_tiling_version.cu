#include "cuda_header.cuh"

__global__ void convolution2D_for_tiling(const float* input, const float* kernel, float* output,
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

int main_tiling(int dim, float* input, float* filter)
{
    int width = dim;
    int height = dim;

    // Allocate host memory for input image, filter, and output
    float* h_input = input;
    float* h_filter = filter;
    float* h_output_tiling = new float[width * height];

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    size_t sharedMemorySize = (blockDim.x + 2 * FILTER_RADIUS) * (blockDim.y + 2 * FILTER_RADIUS) * sizeof(float);

    convolution2D_for_tiling<<<gridDim, blockDim, sharedMemorySize>>>(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS});
    cudaMemcpy(h_output_tiling, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[ ";
    for (int i = 0; i < 10; ++i) std::cout << h_output_tiling[i] << " ";
    std::cout << "]\n";

    delete[] h_output_tiling;

    return 0;
}