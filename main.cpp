#include <string>

#include "src/cuda-impl/cuda_header.cuh"
#include "src/cpp-impl/cpp_header.h"
#include "src/shared/s_header.h"

int main(int argc, char** argv)
{
    if (check_devices_status() == 1) return 1;

    checkOpenMPStatus();

    int value = 0;
    std::cout << "\nEnter size: ";
    std::cin >> value;

    std::cout << "[Generator]:: initializing input and filter\n";
    auto [input, filter] = generate(value);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test]:: cpu convolution\n";
    //TODO mpi multithreading to boost the performances of cpu-version
    assignment_main(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test]:: basic convolution\n";
    main_basic(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test]:: tiling convolution\n";
    main_tiling(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test]:: streams convolution\n";
    main_streams(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test]:: streams + tiling convolution\n";
    main_tiling_streams(value, input, filter);

    delete[] input;
    delete[] filter;

    return 0;
}
