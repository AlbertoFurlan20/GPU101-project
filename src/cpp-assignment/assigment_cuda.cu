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
    for (size_t i = 0; i < array->size(); ++i)
    {
        std::cout << array->operator[](i) << " ";
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
                              int3 inputSize, int3 kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int radiusX = kernelSize.x / 2;
    int radiusY = kernelSize.y / 2;
    int radiusZ = kernelSize.z / 2;

    int outX = max(1, inputSize.x - kernelSize.x + 1);
    int outY = max(1, inputSize.y - kernelSize.y + 1);
    int outZ = max(1, inputSize.z - kernelSize.z + 1);

    if (x < outX && y < outY && z < outZ)
    {
        float result = 0.0f;

        for (int dkx = -radiusX; dkx <= radiusX; ++dkx)
        {
            for (int dky = -radiusY; dky <= radiusY; ++dky)
            {
                for (int dkz = -radiusZ; dkz <= radiusZ; ++dkz)
                {
                    int ix = x + dkx + radiusX;
                    int iy = y + dky + radiusY;
                    int iz = z + dkz + radiusZ;

                    // Handle boundary conditions
                    if (ix >= 0 && ix < inputSize.x && iy >= 0 && iy < inputSize.y && iz >= 0 && iz < inputSize.z)
                    {
                        // #define IDX_3D(x, y, z, width, height) ((z) * (width) * (height) + (y) * (width) + (x))
                        int kernelIndex = (dkz + radiusZ) * kernelSize.y * kernelSize.x +
                            (dky + radiusY) * kernelSize.x
                            + (dkx + radiusX);
                        int inputIndex = iz * inputSize.x * inputSize.y + iy * inputSize.x + ix;

                        result += input[inputIndex] * kernel[kernelIndex];
                    }
                }
            }
        }

        output[z * outX * outY + y * outX + x] = result;
    }
}

__global__ void convolution3D_method2(const float* input, const float* filter, float* output, int3 inputSize,
                                      int3 filterSize)
{
    // 1. calculate thread-grid ref indexes
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int kernelRadiusX = filterSize.x / 2;
    int kernelRadiusY = filterSize.y / 2;
    int kernelRadiusZ = filterSize.z / 2;

    // printf("Thread (%d, %d, %d)\n", x, y, z);  // Debugging thread indices

    if (x < inputSize.x && y < inputSize.y && z < inputSize.z)
    {
        float result = 0.0f;

        for (int kz = 0; kz < filterSize.z; ++kz)
        {
            for (int ky = 0; ky < filterSize.y; ++ky)
            {
                for (int kx = 0; kx < filterSize.x; ++kx)
                {
                    // 1. compute base-level indexes
                    int nx = x + kx - kernelRadiusX;
                    int ny = y + ky - kernelRadiusY;
                    int nz = z + kz - kernelRadiusZ;

                    // Debugging: print input/output index calculations
                    int input_idx = IDX_3D(nx, ny, nz, inputSize.x, inputSize.y);
                    int kernel_idx = IDX_3D(kx, ky, kz, filterSize.x, filterSize.y);


                    if (nx >= 0 && nx < inputSize.x && ny >= 0 && ny < inputSize.y && nz >= 0 && nz < inputSize.z)
                    {
                        // printf("nx=%d, ny=%d, nz=%d, input_idx=%d, kernel_idx=%d\n", nx, ny, nz, input_idx, kernel_idx);
                        result += input[input_idx] * filter[kernel_idx];
                    }
                }
            }
        }

        int outputIdx = IDX_3D(x, y, z, inputSize.x, inputSize.y);
        output[outputIdx] = result;
    }
}


std::pair<dim3, dim3> setSizeAndGrid(unsigned int dim, int3 inputSize)
{
    dim3 blockDim(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    if (dim == 1)
    {
        blockDim.x = 256;
        gridSize.x = (inputSize.x + blockDim.x - 1) / blockDim.x;
    }

    else if (dim == 2)
    {
        blockDim.x = 16;
        blockDim.y = 16;

        gridSize.x = (inputSize.x + blockDim.x - 1) / blockDim.x;
        gridSize.y = (inputSize.y + blockDim.y - 1) / blockDim.y;
    }
    else
    {
        // dim max should be 3 TODO check
        blockDim.x = 8;
        blockDim.y = 8;
        blockDim.z = 8;

        gridSize.x = (inputSize.x + blockDim.x - 1) / blockDim.x;
        gridSize.y = (inputSize.y + blockDim.y - 1) / blockDim.y;
        gridSize.z = (inputSize.z + blockDim.z - 1) / blockDim.z;
    }

    return std::make_pair(blockDim, gridSize);
}

int run_assignment_cuda(int dim)
{
    int width = dim;
    int height = dim == 2 ? dim : 1;
    int depth = dim == 3 ? dim : 1;

    // input_type *input = new input_type[width * height];               // Input
    // filter_type *filter = new filter_type[FILTER_SIZE * FILTER_SIZE]; // Convolution filter

    // Randomly initialize the inputs
    // [NOTE] - DynamicArray<float>* input = new DynamicArray<float>(width * height);  // this is copy init
    //        - DynamicArray<float> input(width * height);                             // this is direct init
    // [NOTE] - use "new" returns a pointer

    int raw_size = width * height * depth;

    auto input = new DynamicArray<float>(raw_size);
    auto output_gpu = new DynamicArray<float>(raw_size); // Output (GPU)
    auto filter = new DynamicArray<float>(FILTER_SIZE * FILTER_SIZE);

    filter->init();
    input->init();

    std::cout << "[INPUT]\n";
    printDynamicArray(input);
    std::cout << std::endl;

    std::cout << "[FILTER]\n";
    printDynamicArray(filter);
    std::cout << std::endl;

    assert(output_gpu->size() == input->size());

    // sizes checks

    // 1. memory allocation
    float* d_filter;
    float* d_input;
    float* d_output;

    // 1.1 allocation
    checkCudaErrors(cudaMalloc(&d_filter, filter->size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_input, input->size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, output_gpu->size() * sizeof(float)));

    // 1.2 passing
    checkCudaErrors(cudaMemcpy(d_input, input->getData(), input->size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, filter->getData(), filter->size() * sizeof(float), cudaMemcpyHostToDevice));

    // 2. gridsize & blocksize setup
    // - v1: let's switch between dim == 1 or 2 or 3
    const int3 inputSize = {width, height, depth};
    constexpr int3 filterSize = {FILTER_SIZE,FILTER_SIZE,FILTER_SIZE};

    auto [fst, snd] = setSizeAndGrid(dim, inputSize);

    // 3. kernel launch
    convolution3D<<<fst, snd>>>(d_input, d_filter, d_output, inputSize, filterSize);
    // convolution3D_method2<<<fst, snd>>>(d_input, d_filter, d_output, inputSize, filterSize);

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
