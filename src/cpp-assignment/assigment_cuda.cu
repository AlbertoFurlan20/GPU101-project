#include <array>
#include <iostream>

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

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
void convolution3D(const DynamicArray<float>* input, const DynamicArray<float>* filter, DynamicArray<float>* output,
                 const unsigned int width, const unsigned int height, const unsigned int depth, const int filterSize)
{
    // indexes:
    int x = blockIdx.x + blockDim.x + threadIdx.x;
    int y = blockIdx.y + blockDim.y + threadIdx.y;
    int z = blockIdx.z + blockDim.z + threadIdx.z;

    int outputCols = height - filterSize + 1;
    int outputRows = width - filterSize + 1;
    int outputDepth = depth - filterSize + 1;

    if (x < outputCols && y < outputRows && z < outputDepth)
    {
        float sum = 0.0f;

        for (int kd = 0; kd < filterSize; ++kd)
        {
            for (int kr = 0; kr < filterSize; ++kr)
            {
                for (int kc = 0; kc < filterSize; ++kc)
                {
                    int inputIdx = ((z + kd) * width + (y + kr)) * height + (x + kc);
                    int kernelIdx = (kd * filterSize + kr) * filterSize + kc;

                    float a = *input[inputIdx].getData();
                    float b = *filter[kernelIdx].getData();

                    sum += a * b;
                }
            }
        }

        int outputIdx = (z * outputRows + y) * outputCols + x;
        output[outputIdx].setData(sum);
    }
}

std::pair<dim3, dim3> setSizeAndGrid(unsigned int dim, unsigned int width, unsigned int height, unsigned int depth)
{
    dim3 blockDim(1, 1, 1);
    dim3 gridSize(1, 1, 1);

    if (dim == 1)
    {
        blockDim.x = 256;
        gridSize.x = (width + blockDim.x - 1) / blockDim.x;
    }

    else if (dim == 2)
    {
        blockDim.x = 16;
        blockDim.y = 16;

        gridSize.x = (width + blockDim.x - 1) / blockDim.x;
        gridSize.y = (height + blockDim.y - 1) / blockDim.y;
    }
    else
    {
        // dim max should be 3 TODO check
        blockDim.x = 8;
        blockDim.y = 8;
        blockDim.z = 8;

        gridSize.x = (width + blockDim.x - 1) / blockDim.x;
        gridSize.y = (height + blockDim.y - 1) / blockDim.y;
        gridSize.z = (depth + blockDim.z - 1) / blockDim.z;
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
    const unsigned int width = dim;
    const unsigned int height = dim;
    const unsigned int depth = dim;

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

    //TODO kernel invokation

    // 1. memory allocation
    DynamicArray<float>* d_filter;
    DynamicArray<float>* d_input;
    DynamicArray<float>* d_output;

    // 1.1 allocation
    checkCudaErrors(cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(DynamicArray<float>)));
    checkCudaErrors(cudaMalloc(&d_input, width * height * sizeof(DynamicArray<float>)));
    checkCudaErrors(cudaMalloc(&d_output, width * height * sizeof(DynamicArray<float>)));

    // 1.2 passing
    checkCudaErrors(cudaMemcpy(d_input, input, input->size() * sizeof(DynamicArray<float>), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, filter, filter->size() * sizeof(DynamicArray<float>), cudaMemcpyHostToDevice));

    // 2. gridsize & blocksize setup
    // - v1: let's switch between dim == 1 or 2 or 3
    auto result = setSizeAndGrid(dim, width, height, depth);

    // v2 - make it dynamic sizing

    // 3. kernel launch
    convolution3D<<<result.first, result.second>>>(d_input, d_filter, d_output, width, height, depth, FILTER_SIZE);

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
