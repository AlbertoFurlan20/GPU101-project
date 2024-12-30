#include "../cuda-impl/cuda_implementation.cuh"

using input_type = float;
using filter_type = input_type;

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
        for (size_t i = 0; i < size_; ++i)
            data[i] = static_cast<T>(rand()) / RAND_MAX;
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
        count++;
    }

    std::cout << std::endl << std::endl;
}

void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void convolution2D(const float* input, const float* kernel, float* output,
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
    if (x < width && y < height) {
        sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + sharedX] = input[IDX_2D(x, y, width)];
    } else {
        sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + sharedX] = 0.0f;
    }

    // Load halo regions (neighboring pixels outside the tile)
    if (threadIdx.x < kernelRadius) {
        if (x >= kernelRadius) {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX - kernelRadius)] =
                input[IDX_2D(x - kernelRadius, y, width)];
        } else {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX - kernelRadius)] = 0.0f;
        }

        if (x + tileWidth < width) {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX + tileWidth)] =
                input[IDX_2D(x + tileWidth, y, width)];
        } else {
            sharedIndexes[sharedY * (tileWidth + 2 * kernelRadius) + (sharedX + tileWidth)] = 0.0f;
        }
    }

    if (threadIdx.y < kernelRadius) {
        if (y >= kernelRadius) {
            sharedIndexes[(sharedY - kernelRadius) * (tileWidth + 2 * kernelRadius) + sharedX] =
                input[IDX_2D(x, y - kernelRadius, width)];
        } else {
            sharedIndexes[(sharedY - kernelRadius) * (tileWidth + 2 * kernelRadius) + sharedX] = 0.0f;
        }

        if (y + tileHeight < height) {
            sharedIndexes[(sharedY + tileHeight) * (tileWidth + 2 * kernelRadius) + sharedX] =
                input[IDX_2D(x, y + tileHeight, width)];
        } else {
            sharedIndexes[(sharedY + tileHeight) * (tileWidth + 2 * kernelRadius) + sharedX] = 0.0f;
        }
    }

    __syncthreads(); // Synchronize threads to ensure all data is loaded

    // Apply convolution
    if (x < width && y < height) {
        float partialResult = 0.0f;

        for (int idxY = 0; idxY < kernelSize; ++idxY) {
            for (int idxX = 0; idxX < kernelSize; ++idxX) {
                int sharedInputY = sharedY - kernelRadius + idxY;
                int sharedInputX = sharedX - kernelRadius + idxX;
                partialResult += kernel[idxY * kernelSize + idxX] *
                                 sharedIndexes[sharedInputY * (tileWidth + 2 * kernelRadius) + sharedInputX];
            }
        }

        output[IDX_2D(x, y, width)] = partialResult;
    }
}

std::pair<dim3, dim3> setSizeAndGrid(int convolutionType, std::pair<int, int> inputParams)
{
    dim3 blockDim(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    int3 inputSize = {inputParams.first, inputParams.second};

    if (convolutionType == 1)
    {
        std::cout << "[WARNING]:: 1D convolution not suppored yet!\n\n";
        blockDim.x = 256;
        gridSize.x = (inputSize.x + blockDim.x - 1) / blockDim.x;
    }

    else if (convolutionType == 2)
    {
        blockDim.x = 16;
        blockDim.y = 16;

        gridSize.x = (inputSize.x + blockDim.x - 1) / blockDim.x;
        gridSize.y = (inputSize.y + blockDim.y - 1) / blockDim.y;
    }
    else
    {
        // dim max should be 3 TODO check
        std::cout << "[WARNING]:: 3D convolution not suppored yet!\n\n";
        blockDim.x = 8;
        blockDim.y = 8;
        blockDim.z = 8;

        gridSize.x = (inputSize.x + blockDim.x - 1) / blockDim.x;
        gridSize.y = (inputSize.y + blockDim.y - 1) / blockDim.y;
        gridSize.z = (inputSize.z + blockDim.z - 1) / blockDim.z;
    }

    std::cout << "\n[SETUP]:: set block size :=(" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")";
    std::cout << "\n[SETUP]:: set grid size :=(" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << ")\n";

    return std::make_pair(gridSize, blockDim);
}

int run_assignment_cuda(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Please specify:\n- matrix dimensions (1);\n- convolution type (2)");
        return EXIT_FAILURE;
    }

    unsigned convolutionType;
    const unsigned dim = atoi(argv[1]);

    std::cout<<"dim: "<< dim<<"\n";
    if (argc > 2)
    {
        convolutionType = atoi(argv[2]);
        std::cout << "convolution type: " << convolutionType << "D";
    } else
    {
        convolutionType = 2;
        std::cout << "convolution type: " << convolutionType << "D (defaulted)";
    }

    std::cout<<"[y] - shared memory\n";

    if (convolutionType < 1 || convolutionType > 3)
    {
        std::cout << "\n[ERROR]:: supported convolution: 2D\n";
        return EXIT_FAILURE;
    }
    // up to 3D convolution is supported
    assert(convolutionType == 1 || convolutionType == 2 || convolutionType == 3);
    std::cout << "supported convolution: 2D\n";

    int width = dim;
    int height = dim;

    int raw_size = width * height;

    auto input = new DynamicArray<float>(raw_size);
    auto output_gpu = new DynamicArray<float>(raw_size);
    auto filter = new DynamicArray<float>(FILTER_SIZE * FILTER_SIZE);

    filter->init();
    input->init();

    assert(output_gpu->size() == input->size());

    // 1. memory allocation
    float *d_filter, *d_input, *d_output;

    // 1.1 allocation
    checkCudaErrors(cudaMalloc(&d_filter, filter->size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_input, input->size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, output_gpu->size() * sizeof(float)));

    // 1.2 passing
    checkCudaErrors(cudaMemcpy(d_input, input->getData(), input->size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, filter->getData(), filter->size() * sizeof(float), cudaMemcpyHostToDevice));

    // 2. gridsize & blocksize setup
    auto inputParams = std::make_pair(width, height);

    auto [gridSize, blockDim] = setSizeAndGrid(convolutionType, inputParams);
    auto filterParams = std::make_pair(FILTER_SIZE, FILTER_RADIUS);

    // Timing setup
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start time
    checkCudaErrors(cudaEventRecord(start));

    // 3. kernel launch
    int sharedMemSize = (blockDim.x + 2 * FILTER_RADIUS) * (blockDim.y + 2 * FILTER_RADIUS) * sizeof(float);
    convolution2D<<<gridSize, blockDim, sharedMemSize>>>(d_input, d_filter, d_output, inputParams, filterParams);

    // 4. device synch & mem copy backwards
    checkCudaErrors(cudaDeviceSynchronize());

    // Record the stop time
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate and display elapsed time
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "\nKernel execution time: " << milliseconds << " ms\n";

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(output_gpu->getData(), d_output, output_gpu->size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "[OUTPUT]\n";
    printDynamicArray(output_gpu);

    // Cleanup and deallocate memory
    delete output_gpu;
    delete filter;
    delete input;

    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);

    // Cleanup timing events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
