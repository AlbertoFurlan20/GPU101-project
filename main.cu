#include "src/cuda-impl/cuda_implementation.cuh"

int main(int argc, char** argv)
{
    // check_devices_status();

    run_assignment_cuda(argc, argv);

    return 0;
}