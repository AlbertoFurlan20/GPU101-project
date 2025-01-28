#include <shared_header.h>

template <typename T>
class DynamicArray
{
private:
    T* data;
    size_t size_;

public:
    // Constructor
    DynamicArray(size_t size) : size_(size)
    {
        data = new T[size];
    }

    // Destructor
    ~DynamicArray()
    {
        delete[] data;
    }

    // Prevent copying
    DynamicArray(const DynamicArray&) = delete;
    DynamicArray& operator=(const DynamicArray&) = delete;

    // Allow moving
    DynamicArray(DynamicArray&& other) noexcept
        : data(other.data), size_(other.size_)
    {
        other.data = nullptr;
        other.size_ = 0;
    }

    // Access operators
    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    // Accessors
    size_t size() const { return size_; }
    T* getData() { return data; }
    const T* getData() const { return data; }

    // Initialize data
    void init()
    {
        for (size_t i = 0; i < size_; ++i)
            data[i] = static_cast<T>(rand()) / RAND_MAX;
    }
};

std::pair<float*, float*> generate(int dim)
{
    try
    {
        auto input = new DynamicArray<float>(dim * dim);
        auto filter = new DynamicArray<float>(FILTER_SIZE * FILTER_SIZE);

        filter->init();
        input->init();

        std::cout << "> (input) [ ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << input->operator[](i) << " ";
        }
        std::cout << " ]\n";

        std::cout << "> (filter) [ ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << filter->operator[](i) << " ";
        }
        std::cout << " ]\n";;

        return std::make_pair(input->getData(), filter->getData());
    } catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::make_pair(nullptr, nullptr);
    }
}
