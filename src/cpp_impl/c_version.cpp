#include <cpp_header.h>

void convolution_cpu_omp(const float* input, const float* filter, input_type* output, const int width, const int height,
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

float* assignment_main(const int dim, const float* input, const float* filter)
{
    const unsigned int width = dim;
    const unsigned int height = dim;
    auto output_cpu = new float[width * height];

    convolution_cpu_omp(input, filter, output_cpu, dim, dim, FILTER_SIZE, FILTER_RADIUS);

    return output_cpu;
}
