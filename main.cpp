#include "src/cpp-impl/cpp_implementation.h"
#include <string>


int main(int argc, char** argv)
{

    int value = 0;
    std::cout << "enter size: ";
    std::cin >> value;
    strcpy(argv[1], std::to_string(value).c_str());

    assignment_main(argc, argv);

    return 0;
}