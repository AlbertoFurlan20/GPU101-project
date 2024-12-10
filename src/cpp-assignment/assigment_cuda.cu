#include <array>
#include <iostream>

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

// 3D Indexing Macro
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
void convolution3D(const float* input, const float* filter, float* output,
                   const int3 inputSize, const int3 filterSize)
{
    // shared allocation
    // 1.  shared memory is just this array => each array-tile is the shared memory
    extern __shared__ float sharedIndexes[];

    // 2. calculate the thread index with ref to the global thread in order to access the correct shared memory tile
    auto shared_index = threadIdx.x + threadIdx.y * blockIdx.x + threadIdx.z * blockIdx.x * blockIdx.y;

    // 3. calculate refs to the "global thread": the position you're in the real thread grid
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    auto z = threadIdx.z + blockIdx.z * blockDim.z;

    // 3.5 perform shared memory load w. bounds checking to ensure

    if (x < inputSize.x && y < inputSize.y && z < inputSize.z)
    {
        // this is the allowed access-space
        // 1. perform the indexing on the input through the 3D macro
        auto index = IDX_3D(x, y, z, inputSize.x, inputSize.y);

        // 2. load shared mem. exploiting that index
        sharedIndexes[shared_index] = input[index];
    }
    else
    {
        // this is the NOT allowed space => apply padding with 0.0f
        sharedIndexes[shared_index] = 0.0f;
    }

    // 4. calculate the kernel radius, i.e.: the distance from the center of the kernel (kernel = filter)
    //    - user the .../ 2 so to center it
    //    - why? 'cause for each thread, we consider neighboring elements up to the kernel radius in all directions
    auto kernelRadiusX = filterSize.x / 2;
    auto kernelRadiusY = filterSize.y / 2;
    auto kernelRadiusZ = filterSize.z / 2;

    // 5. add a barrier to ensure ALL threads conclude the setup phase before starting effective computation
    // [NOTE]: up to now you completed setup
    __syncthreads();

    // 6. [EXECUTION] now let's adjust the real convolution exec phase
    float result = 0.0f;

    // you need 3 for loops to slide throughout the kernel
    // 1° [Z-DIMENSION]
    // 2° [Y-DIMENSION]
    // 3° [X-DIMENSION]
    for (int kz = -kernelRadiusZ; kz <= kernelRadiusZ; ++kz)
    {
        for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ++ky)
        {
            for (int kx = -kernelRadiusX; kx <= kernelRadiusX; ++kx)
            {
                // 1. compute the base-level indexes:
                int nx = x + kx, ny = y + ky, nz = z + kz;

                // 2. compute the partial result
                //    - check you're accessing a valid piece of shared mem.
                if (nx >= 0 && nx < inputSize.x && ny >= 0 && ny < inputSize.y && nz >= 0 && nz < inputSize.y)
                {
                    // 1. calculate kernel-ref index:
                    int kernelIdx = IDX_3D(kx + kernelRadiusX, ky + kernelRadiusY, kz + kernelRadiusZ,
                                           filterSize.x, filterSize.y);

                    // 2. calculate input-ref index
                    int inputIdx = IDX_3D(nx, ny, nz, inputSize.x, inputSize.y);

                    // 3. calculate partial result:
                    result += sharedIndexes[shared_index] * filter[kernelIdx];
                }
            }
        }
    }

    // 7. insert the result into the output
    if (z < inputSize.z && y < inputSize.y && x < inputSize.x)
    {
        // 1. calculate the index:
        int index = IDX_3D(x, y, z, inputSize.x, inputSize.y);

        // 2. set value
        output[index] = result;
    }
}

std::pair<std::pair<dim3, dim3>, int> setSizeAndGrid(unsigned int dim, unsigned int width, unsigned int height,
                                                     unsigned int depth)
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

    int sharedMemorySize = blockDim.x * blockDim.y * blockDim.z * sizeof(DynamicArray<float>);

    return std::make_pair(std::make_pair(blockDim, gridSize), sharedMemorySize);
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
    checkCudaErrors(cudaMemcpy(d_input, input->getData(),
                               input->size() * sizeof(float),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_filter, input->getData(),
                               filter->size() * sizeof(float),
                               cudaMemcpyHostToDevice));

    // 2. gridsize & blocksize setup
    // - v1: let's switch between dim == 1 or 2 or 3
    auto result = setSizeAndGrid(dim, width, height, depth);

    // v2 - make it dynamic sizing

    // 3. kernel launch
    int3 inputSize = {width, height, depth};
    int3 filterSize = {FILTER_SIZE, FILTER_SIZE, FILTER_SIZE};

    convolution3D<<<result.first.first, result.first.second, result.second>>>(
        d_input, d_filter, d_output, inputSize, filterSize);

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
