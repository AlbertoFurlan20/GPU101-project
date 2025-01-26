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
#include <cstdlib>
#include <tuple>

int check_devices_status();

int run_assignment_cuda(int, char**);

int main_test(int);

int main_test2(int);

int main_test3(int);

#endif //CUDA_IMPLEMENTATION_CUH
