#include "src/cuda-impl/cuda_implementation.cuh"

int run()
{
    check_devices_status();

    run_main();

    return 0;
}