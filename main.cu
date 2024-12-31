#include "src/cuda-impl/cuda_implementation.cuh"

#include <string>

int main(int argc, char** argv)
{
    int value = 0;
    std::cout << "enter size: ";
    std::cin >> value;

    char** args = new char*[3];  // Space for two arguments

    // Convert the int value to string and then to char*
    std::string str_value = std::to_string(value);
    args[1] = new char[str_value.length() + 1];  // +1 for null terminator
    strcpy(args[1], str_value.c_str());

    args[2] = new char[2];  // "2" plus null terminator
    strcpy(args[2], "2");

    run_assignment_cuda(argc, args);

    for (int i = 1; i < 3; i++)
    {
        delete[] args[i];
    }

    delete[] args;

    return 0;
}