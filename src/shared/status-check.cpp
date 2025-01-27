#include "s_header.h"

int check_devices_status()
{
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-enabled devices found." << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA-enabled devices.\n";

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        std::cout << "Device " << deviceProp.name << "\n";
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes\n";
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << "\n";
        std::cout << "  Max Threads per Dimension (X, Y, Z): "
            << deviceProp.maxThreadsDim[0] << ", "
            << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << "\n";
        std::cout << "  Max Block Dimensions (X, Y, Z): "
            << deviceProp.maxGridSize[0] << ", "
            << deviceProp.maxGridSize[1] << ", "
            << deviceProp.maxGridSize[2] << "\n";
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem << " bytes\n";
        std::cout << "  Warp Size: " << deviceProp.warpSize << "\n";
        std::cout << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << "\n";
        std::cout << "  Max Registers per Block: " << deviceProp.regsPerBlock << "\n";
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    }

    return 0;
}

void checkOpenMPStatus()
{
    int num_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();

    std::cout << " \nOpenMP Status: " << std::endl;
    std::cout << "  Max threads supported by OpenMP: " << num_threads << std::endl;
    std::cout << "  Available processors: " << num_procs << std::endl;

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (thread_id == 0)
        {
            std::cout << "  Number of threads in parallel region: " << omp_get_num_threads() << std::endl;
        }
    }
}
