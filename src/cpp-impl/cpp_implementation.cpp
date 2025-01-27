#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

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

int assignment_main(int dim, float* input, float* filter)
{


    const unsigned int width = dim;
    const unsigned int height = dim;

    // input_type* input = new input_type[width * height]; // Input
    // filter_type* filter = new filter_type[FILTER_SIZE * FILTER_SIZE]; // Convolution filter
    // input_type* output_cpu = new input_type[width * height]; // Output (CPU)
    input_type* output_gpu = new input_type[width * height]; // Output (GPU)

    // float *h_input = input;
    float *output_cpu = new float[width * height];
    // float *h_filter = new float[filterSize * filterSize];
    // float *h_filter = filter;

    // Randomly initialize the inputs
    // for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
    //     filter[i] = static_cast<filter_type>(rand()) / RAND_MAX;
    //
    // for (int i = 0; i < width * height; ++i)
    //     input[i] = static_cast<input_type>(rand()) / RAND_MAX; // Random value between 0 and 1

    // Call CPU convolution
    convolution_cpu(input, filter, output_cpu, dim, dim, FILTER_SIZE, FILTER_RADIUS);

    std::cout << "Output (first 10 values):" << std::endl;
    std::cout << "> (input) [ ";
    for (int i = 0; i < 10; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << " ]\n";
    std::cout << "> (filter) [ ";
    for (int i = 0; i < 10; ++i) {
        std::cout << filter[i] << " ";
    }
    std::cout << " ]\n";
    for (int i = 0; i < 10; i++)
    {
        std::cout << output_cpu[i] << " ";
    }

    // Cleanup and deallocate memory
    // delete[] input;
    // delete[] filter;
    delete[] output_cpu;
    delete[] output_gpu;

    return EXIT_SUCCESS;
}
