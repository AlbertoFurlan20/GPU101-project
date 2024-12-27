#include "src/cuda-impl/cuda_implementation.cuh"

int main()
{
    check_devices_status();

    run_cuda_main();

    return 0;
}