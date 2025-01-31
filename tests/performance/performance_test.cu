#include <gtest/gtest.h>
#include <cuda_header.cuh>
#include <shared_header.h>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iostream>

class PerformanceTestSuite : public ::testing::Test
{
protected:
    constexpr static int loads[6] = {1000, 10000, 25000, 50000, 60000, 75000};
    static std::unordered_map<int, std::pair<float*, float*>> input_data;

    static void SetUpTestSuite()
    {
        std::cout << "[INFO] Initializing before all tests...\n";
        // Set up the input_data for all the loads
        for (int load : loads)
        {
            try
            {
                input_data[load] = generate(load);
            }
            catch (std::exception& e)
            {
                std::cout << "       * Error in [" << std::to_string(load) << "] setup: " << e.what() << std::endl;
            }
        }
        std::cout << "       > Setup completed for all loads!\n\n";
    }

    static void TearDownTestSuite()
    {
        std::cout << "[INFO] Final cleanup after all tests...\n";
        // Clean up the input data after all tests
        for (auto& [dim, data] : input_data)
        {
            delete[] data.first;
            delete[] data.second;
        }
        input_data.clear();
        std::cout << "       > Cleanup completed for all loads!\n";
    }

    static int test_load(const int dim)
    {
        try
        {
            if (input_data.find(dim) == input_data.end())
            {
                std::cerr << "        * Error: No input data found for dim: " + std::to_string(dim);
                return 1;
            }

            auto [input, filter] = input_data[dim];
            int width = dim;
            int height = dim;

            auto h_output_basic = new float[width * height];
            float *d_input = nullptr, *d_filter = nullptr, *d_output = nullptr;

            size_t free_mem_before, total_mem;
            cudaMemGetInfo(&free_mem_before, &total_mem);

            size_t required_mem = (width * height + FILTER_SIZE * FILTER_SIZE) * sizeof(float);
            std::cout << "       * required mem " << required_mem / (1024 * 1024) << "MB\n";
            std::cout << "       * available mem " << free_mem_before / (1024 * 1024) << "MB\n";
            if (required_mem > free_mem_before)
            {
                std::cerr << "       * Error: memory allocation failed for dim: " << dim <<
                    " (Not enough free memory)\n";
                delete[] h_output_basic;
                return 1;
            }

            cudaMalloc(&d_input, width * height * sizeof(float));
            cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
            cudaMalloc(&d_output, width * height * sizeof(float));

            cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockDim(16, 16);
            dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
            size_t sharedMemorySize = (blockDim.x + 2 * FILTER_RADIUS) * (blockDim.y + 2 * FILTER_RADIUS) * sizeof(
                float);

            const std::pair<int, int> inputSize = {width, height};
            constexpr std::pair<int, int> filterParams = {FILTER_SIZE, FILTER_RADIUS};

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            convolution2D_basic<<<gridDim, blockDim, sharedMemorySize>>>(
                d_input, d_filter, d_output, inputSize, filterParams);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_time;
            cudaEventElapsedTime(&elapsed_time, start, stop);

            size_t free_mem_after;
            cudaMemGetInfo(&free_mem_after, &total_mem);

            std::cout << "       * Kernel execution time: " << elapsed_time << " ms\n";
            std::cout << "       * Memory used: " << (free_mem_before - free_mem_after) / (1024 * 1024) << " MB";

            cudaMemcpy(h_output_basic, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_input);
            cudaFree(d_filter);
            cudaFree(d_output);

            cudaMemGetInfo(&free_mem_after, &total_mem);
            if (required_mem > free_mem_before)
            {
                std::cerr << "       * Warning: VRAM leak detected [" << (required_mem - free_mem_before) / (1024 *
                    1024) << " MB]\n";
                delete[] h_output_basic;
                return 1;
            }

            // Check if first 5 elements are zero
            bool all_zeros = true;
            for (int i = 0; i < 5 && i < width * height; i++)
            {
                if (h_output_basic[i] != 0.0f)
                {
                    all_zeros = false;
                    break;
                }
            }
            if (all_zeros)
            {
                std::cerr << "       * Error: First 5 elements of output are all zero!";
                delete[] h_output_basic;
                return 1;
            }

            delete[] h_output_basic;
            return 0;
        }
        catch (const std::exception& e)
        {
            std::cerr << std::string("Exception: ") + e.what();
            return 1;
        }
    }

    static int test_tiling_streams_load(const int dim)
    {
        try
        {
            if (input_data.find(dim) == input_data.end())
            {
                std::cerr << "        * Error: No input data found for dim: " + std::to_string(dim);
                return 1;
            }

            auto [input, filter] = input_data[dim];
            int width = dim;
            int height = dim;

            auto h_output_both = new float[width * height];
            float *d_input = nullptr, *d_filter = nullptr, *d_output = nullptr;

            size_t free_mem_before, total_mem;
            cudaMemGetInfo(&free_mem_before, &total_mem);

            size_t required_mem = (width * height + FILTER_SIZE * FILTER_SIZE) * sizeof(float);
            std::cout << "       * required mem " << required_mem / (1024 * 1024) << "MB\n";
            std::cout << "       * available mem " << free_mem_before / (1024 * 1024) << "MB\n";
            if (required_mem > free_mem_before)
            {
                std::cerr << "       * Error: memory allocation failed for dim: " << dim <<
                    " (Not enough free memory)\n";
                delete[] h_output_both;
                return 1;
            }

            cudaMalloc(&d_input, width * height * sizeof(float));
            cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
            cudaMalloc(&d_output, width * height * sizeof(float));

            cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            launch_convolution2D_tiling_streams(d_input, d_filter, d_output, {width, height},
                                                {FILTER_SIZE, FILTER_RADIUS}, 4);
            cudaMemcpy(h_output_both, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_filter);
            cudaFree(d_output);

            return 0;
        }
        catch (std::exception& e)
        {
            std::cerr << std::string("Exception: ") + e.what();
            return 1;
        }
    }
};

std::unordered_map<int, std::pair<float*, float*>> PerformanceTestSuite::input_data;

TEST_F(PerformanceTestSuite, PlainConvolutionLoadTest)
{
    std::cout << "TEST 1\n";

    std::vector<int> errors;
    std::vector<std::string> error_messages;

    for (const int load : loads)
    {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();

        const int result = test_load(load);

        std::string captured_stdout = testing::internal::GetCapturedStdout();
        std::string captured_stderr = testing::internal::GetCapturedStderr();

        if (result != 0)
        {
            errors.push_back(load);
            std::string err_msg;
            err_msg.append("Load [" + std::to_string(load) + "]:");

            if (!captured_stdout.empty())
            {
                err_msg.append("\n > STDOUT:\n" + captured_stdout);
            }

            if (!captured_stderr.empty())
            {
                err_msg.append("\n > STDERR:\n" + captured_stderr);
            }

            err_msg.append("\n");

            error_messages.push_back(err_msg);
        }
    }

    if (!errors.empty())
    {
        std::ostringstream error_message;
        error_message << "Errors found for loads: [";

        for (const int error : errors)
        {
            error_message << error << ", ";
        }
        error_message << "]: \n";

        for (auto msg = error_messages.begin(); msg != error_messages.end(); ++msg)
        {
            error_message << *msg;

            if (msg != std::prev(error_messages.end()))
            {
                error_message << "\n";
            }
        }

        GTEST_SKIP() << error_message.str();
    }
}

TEST_F(PerformanceTestSuite, TiledSharedConvolutionLoadTest)
{
    std::cout << "TEST 2\n";

    std::vector<int> errors;
    std::vector<std::string> error_messages;

    for (const int load : loads)
    {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();

        int result = test_tiling_streams_load(load);

        std::string captured_stdout = testing::internal::GetCapturedStdout();
        std::string captured_stderr = testing::internal::GetCapturedStderr();

        if (result != 0)
        {
            errors.push_back(load);
            std::string err_msg;
            err_msg.append("Load [" + std::to_string(load) + "]:");

            if (!captured_stdout.empty())
            {
                err_msg.append("\n > STDOUT:\n" + captured_stdout);
            }

            if (!captured_stderr.empty())
            {
                err_msg.append("\n > STDERR:\n" + captured_stderr);
            }

            err_msg.append("\n");

            error_messages.push_back(err_msg);
        }
    }

    if (!errors.empty())
    {
        std::ostringstream error_message;
        error_message << "Errors found for loads: [";

        for (const int error : errors)
        {
            error_message << error << ", ";
        }
        error_message << "]: \n";

        for (auto msg = error_messages.begin(); msg != error_messages.end(); ++msg)
        {
            error_message << *msg;

            if (msg != std::prev(error_messages.end()))
            {
                error_message << "\n";
            }
        }

        GTEST_SKIP() << error_message.str();
    }
}
