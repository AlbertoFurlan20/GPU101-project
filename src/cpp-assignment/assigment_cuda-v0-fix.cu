#include "../cuda-impl/cuda_implementation.cuh"

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

#define IDX_3D(x, y, z, width, height) ((z) * (width) * (height) + (y) * (width) + (x))

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

std::array<int, 124> flatten2DArray(int rows, int cols, int** array2D)
{
    std::array<int, 124> flatArray = {};
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            flatArray[i * cols + j] = array2D[i][j];
        }
    }

    return flatArray;
}

std::array<int, 124> flatten3DArray(int depth, int rows, int cols, int*** array3D)
{
    std::array<int, 124> flatArray = {};

    for (int d = 0; d < depth; ++d)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                flatArray[(d * rows + i) * cols + j] = array3D[d][i][j];
            }
        }
    }

    return flatArray;
}

__global__ void convolution3D(const float* input, const float* kernel, float* output,
                              std::pair<int, int> inputSize, std::pair<int, int> filterParams)
{
    auto kernelRadius = filterParams.second;
    auto kernelSize = filterParams.first;
    auto width = inputSize.first;
    auto height = inputSize.second;

    int outputY = blockIdx.y * blockDim.y + threadIdx.y; // row idx
    int outputX = blockIdx.x * blockDim.x + threadIdx.x; // col idx

    if (outputY < height && outputX < width)
    {
        float partialResult = 0.0f;

        for (int idxY = 0; idxY < kernelSize; ++idxY)
        {
            for (int idxX = 0; idxX < kernelSize; ++idxX)
            {
                int inputY = outputY - kernelRadius + idxY;
                int inputX = outputX - kernelRadius + idxX;

                if (inputY >= 0 && inputY < height && inputX >= 0 && inputX < width)
                {
                    partialResult += kernel[idxY * kernelSize + idxX] * input[inputY * width + inputX];
                }
            }
        }

        output[outputY * width + outputX] = partialResult;
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
        std::cout << "convolution type: " << convolutionType << "D\n";
    }else
    {
        convolutionType = 2;
        std::cout << "convolution type: " << convolutionType << "D (defaulted)\n";
    }

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

    auto [gridSize, blockDim] = setSizeAndGrid(dim, inputParams);
    auto filterParams = std::make_pair(FILTER_SIZE, FILTER_RADIUS);

    // 3. kernel launch
    convolution3D<<<gridSize, blockDim>>>(d_input, d_filter, d_output, inputParams, filterParams);

    // 4. device synch & mem copy backwards
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(output_gpu->getData(), d_output, output_gpu->size() * sizeof(float),
                               cudaMemcpyDeviceToHost));

    std::cout << "[OUTPUT]\n";
    printDynamicArray(output_gpu);

    // Cleanup and deallocate memory
    delete output_gpu;
    delete filter;
    delete input;

    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
