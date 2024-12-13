#include "src/cuda-impl/cuda_implementation.cuh"
#include "src/cpp-assignment/assignment_cuda_fix.cuh"

int main(int argc, char** argv)
{
    // check_devices_status();

    // run_cuda_main();

    run_assignment_cuda(argc, argv);
    // new_main(argc, argv);
    // run_assignment_cuda(2);
    // run_assignment_cuda(3);

    return 0;
}