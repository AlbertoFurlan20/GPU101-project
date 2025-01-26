#include <string>

#include "src/cuda-impl/cuda_implementation.cuh"

int main(int argc, char** argv)
{
    // if (check_devices_status() == 1) return 1;

    int value = 0;
    std::cout << "enter size: ";
    std::cin >> value;

    char** dict = new char*[3]; // Space for two arguments

    // Initialize the first element to nullptr
    dict[0] = nullptr;

    // Convert the int value to string and then to char*
    std::string str_value = std::to_string(value);

    dict[1] = new char[str_value.length() + 1]; // +1 for null terminator
    strcpy(dict[1], str_value.c_str());

    dict[2] = new char[2]; // "2" plus null terminator
    strcpy(dict[2], "2");

    run_assignment_cuda(argc, dict);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";
    main_test(value);
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";
    main_test2(value);

    delete[] dict;

    return 0;
}
