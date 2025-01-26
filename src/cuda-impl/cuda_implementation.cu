#include "cuda_implementation.cuh"

#include <variant>
#include <omp.h> // OpenMP for parallelization
#include <iostream>
#include <random>
#include <vector>

using input_type = float;
using filter_type = input_type;

#define TILE_WIDTH 512
#define TILE_HEIGHT 512

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

#define IDX_2D(x, y, width) ((y) * (width) + (x))

template <typename T>
class DynamicArray
{
private:
    T* data;
    size_t size_;

public:
    // Constructor
    DynamicArray(size_t size) : size_(size)
    {
        data = new T[size];
    }

    // Destructor
    ~DynamicArray()
    {
        delete[] data;
    }

    // Prevent copying
    DynamicArray(const DynamicArray&) = delete;
    DynamicArray& operator=(const DynamicArray&) = delete;

    // Allow moving
    DynamicArray(DynamicArray&& other) noexcept
        : data(other.data), size_(other.size_)
    {
        other.data = nullptr;
        other.size_ = 0;
    }

    // Access operators
    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    // Accessors
    size_t size() const { return size_; }
    T* getData() { return data; }
    const T* getData() const { return data; }

    // Initialize data
    void init()
    {
        // Mersenne Twister random number generator
        std::random_device rd;
        std::mt19937 mt(rd()); // Use mt19937 for faster random number generation
        std::uniform_real_distribution<T> dist(0.0, 100.0);

        // Parallel initialization using OpenMP
#pragma omp parallel
        {
            // Each thread gets its own RNG to avoid contention
            std::mt19937 thread_rng(mt());

#pragma omp for
            for (size_t i = 0; i < size_; ++i)
            {
                data[i] = dist(thread_rng); // Assign random float in [0.0, 1.0]
            }
        }
    }
};

template <typename T>
void printDynamicArray(DynamicArray<T>* array)
{
    int count = 1;
    for (size_t i = 0; i < array->size(); ++i)
    {
        std::cout << array->operator[](i) << " ";

        if (count % 3 == 0) std::cout << "\n";
        if (count == 9)
        {
            std::cout << "\n clamped output\n";
            break;
        }
        count++;
    }

    std::cout << std::endl << std::endl;
}

void checkCudaErrors(cudaError_t err, const char* file = __FILE__, int line = __LINE__)
{
    if (err != cudaSuccess)
    {
        std::cerr << "\nCUDA Error at: " << file << ":" << line << std::endl
            << "* Error code: " << static_cast<int>(err) << std::endl
            << "* Error type: " << cudaGetErrorName(err) << std::endl
            << "* Error description: " << cudaGetErrorString(err) << std::endl;

        // Get last error state
        cudaError_t lastError = cudaGetLastError();
        if (lastError != err)
        {
            std::cerr << "Additional last error: " << cudaGetErrorString(lastError) << std::endl;
        }

        // Ensure all previous operations have completed
        cudaDeviceSynchronize();

        // Reset device to clear any errors
        cudaDeviceReset();

        exit(EXIT_FAILURE);
    }
}

__global__ void singleKernelConvolution2D(const float* input, const float* kernel, float* output,
                                          std::pair<int, int> inputSize, std::pair<int, int> filterParams)
{
    extern __shared__ float sharedIndexes[];

    int kernelRadius = filterParams.second;
    int kernelSize = filterParams.first;
    int width = inputSize.first;
    int height = inputSize.second;

    int tileWidth = blockDim.x; // Width of the tile
    int tileHeight = blockDim.y; // Height of the tile

    int x = blockIdx.x * tileWidth + threadIdx.x; // Global x index
    int y = blockIdx.y * tileHeight + threadIdx.y; // Global y index

    int sharedX = threadIdx.x + kernelRadius; // Offset for shared memory
    int sharedY = threadIdx.y + kernelRadius;

    // Load data into shared memory
    if (x < width && y < height)
    {
        sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + sharedX] = input[IDX_2D(x, y, width)];
    }
    else
    {
        sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + sharedX] = 0.0f;
    }

    // Load halo regions (neighboring pixels outside the tile)
    if (threadIdx.x < kernelRadius)
    {
        if (x >= kernelRadius)
        {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX - kernelRadius)] =
                input[IDX_2D(x - kernelRadius, y, width)];
        }
        else
        {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX - kernelRadius)] = 0.0f;
        }

        if (x + tileWidth < width)
        {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX + tileWidth)] =
                input[IDX_2D(x + tileWidth, y, width)];
        }
        else
        {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX + tileWidth)] = 0.0f;
        }
    }

    if (threadIdx.y < kernelRadius)
    {
        if (y >= kernelRadius)
        {
            sharedIndexes[(sharedY - kernelRadius) * (tileWidth + 2 * kernelRadius) + sharedX] =
                input[IDX_2D(x, y - kernelRadius, width)];
        }
        else
        {
            sharedIndexes[(sharedY - kernelRadius) * (tileWidth + 2 * kernelRadius) + sharedX] = 0.0f;
        }

        if (y + tileHeight < height)
        {
            sharedIndexes[(sharedY + tileHeight) * (tileWidth + 2 * kernelRadius) + sharedX] =
                input[IDX_2D(x, y + tileHeight, width)];
        }
        else
        {
            sharedIndexes[(sharedY + tileHeight) * (tileWidth + 2 * kernelRadius) + sharedX] = 0.0f;
        }
    }

    __syncthreads(); // Synchronize threads to ensure all data is loaded

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
                    sharedIndexes[sharedInputY * (tileWidth + 2 * kernelRadius) + sharedInputX];
            }
        }

        output[IDX_2D(x, y, width)] = partialResult;
    }
}

__global__ void multiKernelConvolution2D(float* input, float* output, int inputWidth, int inputHeight, float* filter,
                                         int filterSize, int startX, int startY)
{
    extern __shared__ float sharedMem[];

    int sharedWidth = blockDim.x + 2 * FILTER_RADIUS;

    // Thread position in the global grid
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // Thread position in shared memory
    int sharedX = threadIdx.x + FILTER_RADIUS;
    int sharedY = threadIdx.y + FILTER_RADIUS;

    // Global indices for this thread
    int globalX = tx + startX;
    int globalY = ty + startY;

    // Initialize shared memory with zeros
    sharedMem[sharedX + sharedY * sharedWidth] = 0.0f;

    // Load the current block into shared memory
    if (globalX < inputWidth && globalY < inputHeight)
    {
        sharedMem[sharedX + sharedY * sharedWidth] = input[IDX_2D(globalX, globalY, inputWidth)];
    }

    // Handle halo regions safely
    if (threadIdx.x < FILTER_RADIUS && (globalX - FILTER_RADIUS) >= 0)
    {
        sharedMem[(sharedX - FILTER_RADIUS) + sharedY * sharedWidth] =
            (globalX - FILTER_RADIUS) < inputWidth && globalY < inputHeight
                ? input[IDX_2D(globalX - FILTER_RADIUS, globalY, inputWidth)]
                : 0.0f;
    }

    if (threadIdx.x >= blockDim.x - FILTER_RADIUS && (globalX + FILTER_RADIUS) < inputWidth)
    {
        sharedMem[(sharedX + FILTER_RADIUS) + sharedY * sharedWidth] =
            (globalX + FILTER_RADIUS) < inputWidth && globalY < inputHeight
                ? input[IDX_2D(globalX + FILTER_RADIUS, globalY, inputWidth)]
                : 0.0f;
    }

    if (threadIdx.y < FILTER_RADIUS && (globalY - FILTER_RADIUS) >= 0)
    {
        sharedMem[sharedX + (sharedY - FILTER_RADIUS) * sharedWidth] =
            globalX < inputWidth && (globalY - FILTER_RADIUS) < inputHeight
                ? input[IDX_2D(globalX, globalY - FILTER_RADIUS, inputWidth)]
                : 0.0f;
    }

    if (threadIdx.y >= blockDim.y - FILTER_RADIUS && (globalY + FILTER_RADIUS) < inputHeight)
    {
        sharedMem[sharedX + (sharedY + FILTER_RADIUS) * sharedWidth] =
            globalX < inputWidth && (globalY + FILTER_RADIUS) < inputHeight
                ? input[IDX_2D(globalX, globalY + FILTER_RADIUS, inputWidth)]
                : 0.0f;
    }

    __syncthreads();

    // Apply convolution if within valid global bounds
    if (globalX < inputWidth && globalY < inputHeight)
    {
        float result = 0.0f;
        for (int fy = -FILTER_RADIUS; fy <= FILTER_RADIUS; fy++)
        {
            for (int fx = -FILTER_RADIUS; fx <= FILTER_RADIUS; fx++)
            {
                int sharedIdx = (sharedX + fx) + (sharedY + fy) * sharedWidth;
                int filterIdx = (fy + FILTER_RADIUS) * filterSize + (fx + FILTER_RADIUS);
                result += sharedMem[sharedIdx] * filter[filterIdx];
            }
        }
        output[IDX_2D(globalX, globalY, inputWidth)] = result;
    }
}

std::pair<dim3, dim3> setSizeAndGrid(int convolutionType, std::pair<int, int> inputParams)
{
    dim3 blockDim(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    int width = inputParams.first;
    int height = inputParams.second;

    // Query device properties for block and grid size constraints
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int maxBlockX = deviceProp.maxThreadsDim[0];
    int maxBlockY = deviceProp.maxThreadsDim[1];
    int maxGridX = deviceProp.maxGridSize[0];
    int maxGridY = deviceProp.maxGridSize[1];
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if (convolutionType == 2)
    {
        // Choose block dimensions (16x16 threads is common for 2D)
        blockDim.x = std::min(16, maxBlockX);
        blockDim.y = std::min(16, maxBlockY);

        // Ensure blockDim does not exceed maxThreadsPerBlock
        if (blockDim.x * blockDim.y > maxThreadsPerBlock)
        {
            blockDim.y = maxThreadsPerBlock / blockDim.x;
        }

        // Calculate grid dimensions
        gridSize.x = (width + blockDim.x - 1) / blockDim.x;
        gridSize.y = (height + blockDim.y - 1) / blockDim.y;

        // Clamp grid dimensions to device constraints
        gridSize.x = std::min(static_cast<int>(gridSize.x), maxGridX);
        gridSize.y = std::min(static_cast<int>(gridSize.y), maxGridY);
    }
    else
    {
        std::cerr << "[WARNING]:: Only 2D convolution is supported!" << std::endl;
    }

    return std::make_pair(gridSize, blockDim);
}

std::tuple<int, int*, int*> getDeviceConstraints()
{
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return std::make_tuple(-1, static_cast<int*>(nullptr), static_cast<int*>(nullptr));
    }

    if (deviceCount > 0)
    {
        // For simplicity, i'll assume 1x device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        auto gridSizeConstraints = new int[3];
        auto blockConstraints = new int[3];

        for (int i = 0; i < 3; i++)
        {
            gridSizeConstraints[i] = deviceProp.maxGridSize[i];
            blockConstraints[i] = deviceProp.maxThreadsDim[i];
        }

        auto sharedMemoryConstraints = deviceProp.sharedMemPerBlock;

        return std::make_tuple(sharedMemoryConstraints, gridSizeConstraints, blockConstraints);
    }

    return std::make_tuple(-1, static_cast<int*>(nullptr), static_cast<int*>(nullptr));
}

std::tuple<int, int, int> parseInput(int argc, char** argv)
{
    std::cout << "[INFO]\n";
    if (argc < 2)
    {
        std::cerr << "Please specify:\n- matrix dimensions (1);\n- convolution type (2)" << std::endl;
        std::exit(1);
    }

    int convolutionType;
    int dim;

    try
    {
        // Convert argv[1] to integer (matrix dimensions)
        dim = std::stoi(argv[1]);
        std::cout << " * dim: " << dim << "\n";

        // Convert argv[2] to integer (convolution type) if provided
        if (argc > 2)
        {
            convolutionType = std::stoi(argv[2]);
            std::cout << " * convolution type: " << convolutionType << "D ";
        }
        else
        {
            convolutionType = 2;
            std::cout << " * convolution type: " << convolutionType << "D (defaulted) ";
        }

        // Validate convolution type
        if (convolutionType < 1 || convolutionType > 3)
        {
            throw std::out_of_range("Supported convolution types are 1D, 2D, and 3D.");
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing input: " << e.what() << std::endl;
        std::exit(1);
    }

    std::cout << "[y] - shared memory\n";
    std::cout << " * supported convolution: " << convolutionType << "D\n";

    return std::make_tuple(dim, dim, convolutionType);
}

std::tuple<DynamicArray<float>*, DynamicArray<float>*, DynamicArray<float>*> initArgs(int width, int height)
{
    try
    {
        int prod = width * height;
        auto input = new DynamicArray<float>(prod);
        auto output = new DynamicArray<float>(prod);
        auto filter = new DynamicArray<float>(FILTER_SIZE * FILTER_SIZE);

        filter->init();
        input->init();

        return {input, output, filter};
    }
    catch (std::exception& e)
    {
        long long prod = static_cast<long long>(width) * height;
        auto input = new DynamicArray<float>(prod);
        auto output = new DynamicArray<float>(prod);
        auto filter = new DynamicArray<float>(FILTER_SIZE * FILTER_SIZE);

        filter->init();
        input->init();

        return {input, output, filter};
    }
}

std::tuple<float*, float*, float*> allocateAndInitDeviceMemory(DynamicArray<float>* filter, DynamicArray<float>* input,
                                                               DynamicArray<float>* output)
{
    float *d_filter, *d_input, *d_output;

    checkCudaErrors(cudaMalloc(&d_filter, filter->size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_input, input->size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, output->size() * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_input, input->getData(), input->size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, filter->getData(), filter->size() * sizeof(float), cudaMemcpyHostToDevice));

    return {d_filter, d_input, d_output};
}

bool performComputationalFeasibility(std::tuple<int, int> sharedMemoryParams, std::tuple<int*, dim3> gridSizeParams,
                                     std::tuple<int*, dim3> blockSizeParams)
{
    auto [sharedMax, sharedAmount] = sharedMemoryParams;
    auto [gridMax, gridAmount] = gridSizeParams;
    auto [blockMax, blockAmount] = blockSizeParams;

    // shared memory check
    if (sharedMax < sharedAmount)
    {
        std::cout << "[ERROR]:: shared memory constraint violation\n";
        std::cout << "          * [max]\t!<\t[yours]\n";
        std::cout << "          * " << sharedMax << " !< " << sharedAmount << "\n";
        return false;
    }

    if (gridMax[0] < gridAmount.x * blockAmount.x || gridMax[1] < gridAmount.y * blockAmount.y || gridMax[2] <
        gridAmount.z * blockAmount.z)
    {
        std::cout << "[ERROR]:: grid size constraint violation\n";
        std::cout << "          * [max]\t!<\t[yours]\n";
        std::cout << "          * " << gridMax[0] << " !< " << gridAmount.x * blockAmount.x << "\n";
        std::cout << "          * " << gridMax[1] << " !< " << gridAmount.y * blockAmount.y << "\n";
        std::cout << "          * " << gridMax[2] << " !< " << gridAmount.z * blockAmount.z << "\n";
        return false;
    }

    if (blockMax[0] < blockAmount.x || blockMax[1] < blockAmount.y || blockMax[2] < blockAmount.z)
    {
        std::cout << "[ERROR]:: block size constraint violation\n";
        std::cout << "          * [max]\t!<\t[yours]\n";
        std::cout << "          * " << blockMax[0] << " !< " << blockAmount.x << "\n";
        std::cout << "          * " << blockMax[1] << " !< " << blockAmount.y << "\n";
        std::cout << "          * " << blockMax[2] << " !< " << blockAmount.z << "\n";

        return false;
    }

    return true;
}

std::variant<bool, std::tuple<dim3, dim3, std::pair<int, int>, std::pair<int, int>, int>> runSizeSetup(
    int width, int height, int convolutionType)
{
    auto inputParams = std::make_pair(width, height);
    auto filterParams = std::make_pair(FILTER_SIZE, FILTER_RADIUS);

    auto [gridSize, blockDim] = setSizeAndGrid(convolutionType, inputParams);

    int sharedMemSize = (blockDim.x + 2 * FILTER_RADIUS) * (blockDim.y + 2 * FILTER_RADIUS) * sizeof(float);

    auto [sharedMemConstraint, gridSizeConstraint, blockSizeConstraint] = getDeviceConstraints();

    std::cout << "\n[SETUP]:: set block size :=(" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")";
    std::cout << "\n          max size:=(" << blockSizeConstraint[0] << ", " << blockSizeConstraint[1] << ", " <<
        blockSizeConstraint[2] << ")";

    std::cout << "\n[SETUP]:: set grid size :=(" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << ")";
    std::cout << "\n          max size:=(" << gridSizeConstraint[0] << ", " << gridSizeConstraint[1] << ", " <<
        gridSizeConstraint[2] << ")";

    std::cout << "\n[SETUP]:: set shared mem size:= " << sharedMemSize;
    std::cout << "\n          max shared mem:= " << sharedMemConstraint << std::endl;

    if (!performComputationalFeasibility({sharedMemConstraint, sharedMemSize},
                                         {gridSizeConstraint, gridSize},
                                         {blockSizeConstraint, blockDim}))
    {
        return false;
    }

    return std::make_tuple(gridSize, blockDim, inputParams, filterParams, sharedMemSize);
}

template <typename T>
float launchSingleKernel(std::variant<bool, T> outcome, float* d_input, float* d_filter, float* d_output)
{
    auto [gridSize, blockDim, inputParams, filterParams, sharedMemSize] = std::get<std::tuple<
        dim3, dim3, std::pair<int, int>, std::pair<int, int>, int>>(outcome);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // 3. kernel launch
    // Record the start time
    checkCudaErrors(cudaEventRecord(start));

    singleKernelConvolution2D<<<gridSize, blockDim, sharedMemSize>>>(d_input, d_filter, d_output, inputParams,
                                                                     filterParams);

    // 4. device synch & mem copy backwards
    checkCudaErrors(cudaDeviceSynchronize());

    // Record the stop time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate and display elapsed time
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    // Cleanup timing events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return milliseconds;
}

float launchStream(float* d_input, float* d_filter, float* d_output, std::pair<int, int> inputSize)
{
    int tileWidth = TILE_WIDTH;
    int tileHeight = TILE_HEIGHT;

    int inputWidth = inputSize.first;
    int inputHeight = inputSize.second;

    int numTilesX = (inputWidth + tileWidth - 1) / tileWidth;
    int numTilesY = (inputHeight + tileHeight - 1) / tileHeight;

    int numStreams = 4;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    int streamIdx = 0;
    int sharedMemSize = (tileWidth + 2 * FILTER_RADIUS) * (tileHeight + 2 * FILTER_RADIUS) * sizeof(float);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int ty = 0; ty < numTilesY; ++ty)
    {
        for (int tx = 0; tx < numTilesX; ++tx)
        {
            int startX = tx * tileWidth;
            int startY = ty * tileHeight;

            dim3 blockSize(16, 16, 1);
            dim3 gridSize((tileWidth + blockSize.x - 1) / blockSize.x, (tileHeight + blockSize.y - 1) / blockSize.y,1);

            multiKernelConvolution2D<<<gridSize, blockSize, sharedMemSize, streams[streamIdx]>>>(
                d_input, d_output, inputWidth, inputHeight, d_filter, FILTER_SIZE, startX, startY);

            streamIdx = (streamIdx + 1) % numStreams;
        }
    }

    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup streams
    for (int i = 0; i < numStreams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int run_assignment_cuda(int argc, char** argv)
{
    auto [width, height, convolutionType] = parseInput(argc, argv);

    std::cout << "[RUNNING INIT]:: startup ...\n";
    auto [input, output, filter] = initArgs(width, height);
    std::cout << "[RUNNING INIT]:: done ...\n";

    assert(output->size() == input->size());

    // 1. memory allocation
    auto [d_filter, d_input, d_output] = allocateAndInitDeviceMemory(filter, input, output);

    // 2. gridsize & blocksize setup
    auto outcome = runSizeSetup(width, height, convolutionType);

    float milliseconds = 0.0f;

    if (std::holds_alternative<bool>(outcome))
    {
        std::cout << "[MODE]:: toggled stream mode\n";
        milliseconds = launchStream(d_input, d_filter, d_output, {width, height});
    }
    else
    {
        std::cout << "[MODE]:: toggled single kernel mode\n";
        milliseconds = launchSingleKernel(outcome, d_input, d_filter, d_output);
    }

    std::cout << "\nKernel execution time: " << milliseconds << " ms\n";

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(output->getData(), d_output, output->size() * sizeof(float),
                               cudaMemcpyDeviceToHost));

    std::cout << "\n[OUTPUT]\n";
    printDynamicArray(output);

    // Cleanup and deallocate memory
    delete output;
    delete filter;
    delete input;

    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
