#ifndef CPP_IMPLEMENTATION_H
#define CPP_IMPLEMENTATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

using input_type = float;
using filter_type = input_type;

float* assignment_main(int, const float*, const float*);

#endif //CPP_IMPLEMENTATION_H
