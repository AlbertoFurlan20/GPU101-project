#include <shared_header.h>

void log_array(const float* array)
{
    std::cout << "[ ";

    for (int i = 0; i < 10; ++i)
    {
        std::cout << array[i] << " ";
    }

    std::cout << "]\n\n";
}
