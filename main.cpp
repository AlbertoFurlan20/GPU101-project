#include <cuda_header.cuh>
#include <cpp_header.h>
#include <shared_header.h>

int main(int argc, char** argv)
{
    if (check_devices_status() == 1) return 1;

    checkOpenMPStatus();

    int value = 0;
    std::cout << "\nEnter size: ";
    std::cin >> value;

    std::cout << "[Generator]:: initializing input and filter\n";
    try
    {
        auto [input, filter] = generate(value);

        std::cout << "\n";
        std::cout << "\n";

        std::cout << "[Test]:: cpu convolution\n";
//        auto output = assignment_main(value, input, filter);
//        log_array(output);
//        delete[] output;
//        std::cout << "\n";
//        std::cout << "\n";
//        std::cout << "[Test]:: basic convolution\n";
//        output = main_basic(value, input, filter);
//        log_array(output);
//        delete[] output;
//        std::cout << "\n";
//        std::cout << "\n";
//        std::cout << "[Test]:: tiling convolution\n";
//        output = main_tiling(value, input, filter);
//        log_array(output);
//        delete[] output;
//        std::cout << "\n";
//        std::cout << "\n";
//        std::cout << "[Test]:: streams convolution\n";
//        output = main_streams(value, input, filter);
//        log_array(output);
//        delete[] output;
//        std::cout << "\n";
//        std::cout << "\n";
//        std::cout << "[Test]:: streams + tiling convolution\n";
        auto output = main_tiling_streams(value, input, filter);
        log_array(output);
        delete[] output;

//        delete[] input;
//        delete[] filter;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: input/filter initialization failed\n";
        return 1;
    }
}
