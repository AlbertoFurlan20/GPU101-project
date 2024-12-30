#include "cuda_implementation.cuh"

int check_devices_status() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled devices found." << std::endl;
        return 1;
    }

    std::cout << "CUDA-enabled device count: " << deviceCount << std::endl;

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        std::cout << "Device " << deviceId << " properties:\n";
        std::cout << "  Name: " << deviceProp.name << "\n";
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
