#include "cpp_implementation.h"

using namespace std;

vector<vector<double>> convolve2D(const vector<vector<double>>& input,
                                  const vector<vector<double>>& kernel,
                                  int stride = 1, int padding = 0) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    int outputRows = (inputRows + 2 * padding - kernelRows) / stride + 1;
    int outputCols = (inputCols + 2 * padding - kernelCols) / stride + 1;

    vector<vector<double>> paddedInput(inputRows + 2 * padding,
                                       vector<double>(inputCols + 2 * padding, 0));
    vector<vector<double>> output(outputRows, vector<double>(outputCols, 0));

    // Add padding to input
    for (int i = 0; i < inputRows; ++i)
        for (int j = 0; j < inputCols; ++j)
            paddedInput[i + padding][j + padding] = input[i][j];

    // Convolution
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            double sum = 0.0;
            for (int m = 0; m < kernelRows; ++m) {
                for (int n = 0; n < kernelCols; ++n) {
                    sum += kernel[m][n] * paddedInput[i * stride + m][j * stride + n];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

int run_main() {
    vector<vector<double>> input = {{1, 2, 3},
                                    {4, 5, 6},
                                    {7, 8, 9}};
    vector<vector<double>> kernel = {{-1, -2, -1},
                                      {0,  0,  0},
                                      {1,  2,  1}};
    vector<vector<double>> output = convolve2D(input, kernel);

    for (const auto& row : output) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
    return 0;
}
