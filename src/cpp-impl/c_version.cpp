#include "cpp_header.h"

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

void convolution_cpu(float* input, const float* filter, input_type* output, const int width, const int height,
                     const int filter_size, const int filter_radius)
{
    for (int outRow = 0; outRow < width; outRow++)
    {
        for (int outCol = 0; outCol < height; outCol++)
        {
            input_type value{0.0f};
            for (int row = 0; row < filter_size; row++)
                for (int col = 0; col < filter_size; col++)
                {
                    int inRow = outRow - filter_radius + row;
                    int inCol = outCol - filter_radius + col;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        value += filter[row * filter_size + col] * input[inRow * width + inCol];
                    }
                }
            output[outRow * width + outCol] = value;
        }
    }
}

void convolution_cpu_omp(float* input, const float* filter, input_type* output, const int width, const int height,
                     const int filter_size, const int filter_radius)
{
    // Parallelize the outer two loops using OpenMP
#pragma omp parallel for collapse(2)
    for (int outRow = 0; outRow < width; outRow++)
    {
        for (int outCol = 0; outCol < height; outCol++)
        {
            input_type value{0.0f};
            for (int row = 0; row < filter_size; row++)
                for (int col = 0; col < filter_size; col++)
                {
                    int inRow = outRow - filter_radius + row;
                    int inCol = outCol - filter_radius + col;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        value += filter[row * filter_size + col] * input[inRow * width + inCol];
                    }
                }
            output[outRow * width + outCol] = value;
        }
    }
}

int assignment_main(int dim, float* input, float* filter)
{
    const unsigned int width = dim;
    const unsigned int height = dim;
    float *output_cpu = new float[width * height];

    // Call CPU convolution
    // convolution_cpu(input, filter, output_cpu, dim, dim, FILTER_SIZE, FILTER_RADIUS);
    convolution_cpu_omp(input, filter, output_cpu, dim, dim, FILTER_SIZE, FILTER_RADIUS);

    std::cout << "[ ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << output_cpu[i] << " ";
    }
    std::cout << " ]\n";

    delete[] output_cpu;

    return EXIT_SUCCESS;
}
