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

float* main_basic(int, const float*, const float*);

__global__ void convolution2D_basic(const float*, const float*, float*, std::pair<int, int>, std::pair<int, int>);

void launch_convolution2D_tiling_streams(const float* , const float* , float* ,
                                         std::pair<int, int> , std::pair<int, int>,
                                         int );

float* main_streams(int, const float*, const float*);

float* main_tiling(int, const float*, const float*);

float* main_tiling_streams(int, const float*, const float*);

#endif //CUDA_IMPLEMENTATION_CUH
