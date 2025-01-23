#include "src/cuda-impl/cuda_implementation.cuh"

int main(int argc, char** argv)
{
    if (check_devices_status() == 1) return 1;

    run_assignment_cuda(argc, argv);

    return 0;
}
