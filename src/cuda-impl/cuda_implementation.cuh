//
// Created by Alberto Furlan on 04/12/24.
//

#ifndef CUDA_IMPLEMENTATION_CUH
#define CUDA_IMPLEMENTATION_CUH

#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include <array>
#include <cassert>
#include <iostream>

int check_devices_status();

int run_cuda_main();

int run_assignment_cuda(int);

#endif //CUDA_IMPLEMENTATION_CUH
