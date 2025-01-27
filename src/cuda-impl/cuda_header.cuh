#ifndef CUDA_IMPLEMENTATION_CUH
#define CUDA_IMPLEMENTATION_CUH

#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cstdlib>
#include <tuple>
#include "cuda_header.cuh"
#include <cuda_runtime.h>

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)
#define TILE_WIDTH 32

#define IDX_2D(x, y, width) ((y) * (width) + (x))

std::pair<float*, float*> run_assignment_cuda(int, char**);

int main_basic(int, const float*, const float* );

int main_streams(int, const float*input, const float* );

int main_tiling(int, const float*input, const float* );

int main_tiling_streams(int, const float*input, const float* );

#endif //CUDA_IMPLEMENTATION_CUH
