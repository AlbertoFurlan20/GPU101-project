# include "cuda_header.cuh"

__global__ void convolution2D_for_tiling_streams(const float* input, const float* kernel, float* output,
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

        convolution2D_for_tiling_streams<<<gridDim, blockDim, (blockDim.x + 2 * filterParams.second) * (blockDim.y + 2 *
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

int main_tiling_streams(const int dim, const float* input, const float* filter)
{
    int width = dim;
    int height = dim;

    const auto h_input = input;
    const auto h_filter = filter;
    const auto h_output_both = new float[width * height];

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    launch_convolution2D_tiling_streams(d_input, d_filter, d_output, {width, height}, {FILTER_SIZE, FILTER_RADIUS}, 4);
    cudaMemcpy(h_output_both, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[ ";
    for (int i = 0; i < 10; ++i) std::cout << h_output_both[i] << " ";
    std::cout << "]\n";

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    delete[] h_output_both;

    return 0;
}