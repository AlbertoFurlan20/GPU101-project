#include <array>
#include <iostream>

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
    DynamicArray(size_t size) : size_(size)
    {
        data = new T[size];
    }

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

    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }
    size_t size() const { return size_; }
    T* getData() { return data; }
    const T* getData() const { return data; }
    void setData(T newData) { data = newData; }

    void init()
    {
        for (int i = 0; i < size_ * size_; ++i)
            data[i] = static_cast<T>(rand()) / RAND_MAX;
    }
};

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

__global__
void convolution3D(const float* input, const float* filter, float* output, int3 inputSize, int3 filterSize)
{
    // 1. calculate thread-grid ref indexes
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int kernelRadiusX = filterSize.x / 2;
    int kernelRadiusY = filterSize.y / 2;
    int kernelRadiusZ = filterSize.z / 2;

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
                    int nx = x + kx, ny = y + ky, nz = z + kz;

                    if (nx >= 0 && nx < inputSize.x && ny >= 0 && ny < inputSize.y && nz >= 0 && nz < inputSize.z)
                    {
                        int kernel_idx = IDX_3D(kx + kernelRadiusX, ky + kernelRadiusY, kz, + kernelRadiusZ,
                                                filterSize.x, filterSize.y);

                        int input_idx = IDX_3D(nx, ny, nz, inputSize.x, inputSize.y);

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

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Please specify matrix dimensions\n");
        return EXIT_FAILURE;
    }

    const unsigned dim = atoi(argv[1]);
    const int width = dim;
    const int height = dim;
    const int depth = dim;

    // input_type *input = new input_type[width * height];               // Input
    // filter_type *filter = new filter_type[FILTER_SIZE * FILTER_SIZE]; // Convolution filter

    // Randomly initialize the inputs
    // [NOTE] - DynamicArray<float>* input = new DynamicArray<float>(width * height);  // this is copy init
    //        - DynamicArray<float> input(width * height);                             // this is direct init
    // [NOTE] - use "new" returns a pointer

    auto output_gpu = new DynamicArray<float>(width * height); // Output (GPU)
    auto filter = new DynamicArray<float>(FILTER_SIZE * FILTER_SIZE);
    auto input = new DynamicArray<float>(width * height);

    filter->init();
    input->init();

    assert(output_gpu->size() == input->size());

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

    // 4. device synch & mem copy backwards
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(output_gpu, d_output, output_gpu->size(), cudaMemcpyDeviceToHost));

    // Cleanup and deallocate memory
    delete[] output_gpu;
    delete[] filter;
    delete[] input;

    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
