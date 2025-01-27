#include <string>

#include "src/cuda-impl/cuda_implementation.cuh"
#include "src/cpp-impl/cpp_implementation.h"

int main(int argc, char** argv)
{
    // if (check_devices_status() == 1) return 1;

    int value = 0;
    std::cout << "enter size: ";
    std::cin >> value;

    char** dict = new char*[3];

    dict[0] = nullptr;

    std::string str_value = std::to_string(value);

    dict[1] = new char[str_value.length() + 1];
    strcpy(dict[1], str_value.c_str());

    dict[2] = new char[2];
    strcpy(dict[2], "2");

    auto [input, filter] = run_assignment_cuda(argc, dict);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test bench]\n";
    assignment_main(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test 1]\n";
    main_test(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test 2]\n";
    main_test2(value, input, filter);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "[Test 3]\n";
    main_test3(value, input, filter);

    delete[] dict;
    delete[] input;
    delete[] filter;

    return 0;
}
