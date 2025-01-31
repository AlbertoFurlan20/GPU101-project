//
// Created by Alberto Furlan on 27/01/25.
//

#ifndef S_HEADER_H
#define S_HEADER_H

#include "cuda_runtime.h"
#include <iostream>
#include <cstdlib>
#include <omp.h>

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

int check_devices_status();

void checkOpenMPStatus();

std::pair<float*, float*> generate(int);

void log_array(const float*);

#endif //S_HEADER_H
