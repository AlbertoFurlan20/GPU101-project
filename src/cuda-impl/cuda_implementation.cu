//
// Created by Alberto Furlan on 04/12/24.
//

#include "cuda_implementation.cuh"

__global__ void convolve2D(const double* input, const double* kernel, double* output,
                           int inputRows, int inputCols,
                           int kernelRows, int kernelCols,
                           int stride, int outputRows, int outputCols) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < outputRows && outCol < outputCols) {
        double sum = 0.0;
        for (int m = 0; m < kernelRows; ++m) {
            for (int n = 0; n < kernelCols; ++n) {
                int inRow = outRow * stride + m;
                int inCol = outCol * stride + n;
                if (inRow < inputRows && inCol < inputCols) {
                    sum += input[inRow * inputCols + inCol] * kernel[m * kernelCols + n];
                }
            }
        }
        output[outRow * outputCols + outCol] = sum;
    }
}

void runConvolution(const std::vector<std::vector<double>>& input,
                    const std::vector<std::vector<double>>& kernel) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();
    int stride = 1;
    int outputRows = (inputRows - kernelRows) / stride + 1;
    int outputCols = (inputCols - kernelCols) / stride + 1;

    size_t inputSize = inputRows * inputCols * sizeof(double);
    size_t kernelSize = kernelRows * kernelCols * sizeof(double);
    size_t outputSize = outputRows * outputCols * sizeof(double);

    double *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);

    std::vector<double> flatInput, flatKernel, flatOutput(outputRows * outputCols);
    for (const auto& row : input)
        flatInput.insert(flatInput.end(), row.begin(), row.end());
    for (const auto& row : kernel)
        flatKernel.insert(flatKernel.end(), row.begin(), row.end());

    cudaMemcpy(d_input, flatInput.data(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, flatKernel.data(), kernelSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outputCols + 15) / 16, (outputRows + 15) / 16);
    convolve2D<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output,
                                                   inputRows, inputCols,
                                                   kernelRows, kernelCols,
                                                   stride, outputRows, outputCols);

    cudaMemcpy(flatOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            std::cout << flatOutput[i * outputCols + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    std::vector<std::vector<double>> input = {{1, 2, 3},
                                              {4, 5, 6},
                                              {7, 8, 9}};
    std::vector<std::vector<double>> kernel = {{-1, -2, -1},
                                                {0,  0,  0},
                                                {1,  2,  1}};
    runConvolution(input, kernel);
    return 0;
}