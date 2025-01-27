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

std::pair<float*, float*> run_assignment_cuda(int argc, char** argv);

int main_test(int dim, float* input, float* filter);

int main_test2(int dim, float* input, float* filter);

int main_test3(int, float* input, float* filter);

#endif //CUDA_IMPLEMENTATION_CUH
