#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <string>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

using namespace std;

size_t XorShift128() {
    static size_t x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    size_t t = x ^ (x << 11);
    x = y; y = z; z = w;
    w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    return w % 1000;
}

void save_matrix(const vector<vector<int>>& mat, const fs::path& dir, const string& filename) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    fs::path filepath = dir / (filename + ".txt");
    ofstream file(filepath);
    if (!file) {
        throw runtime_error("Can't open file for writing: " + filepath.string());
    }
    for (const auto& row : mat) {
        for (int v : row) {
            file << v << ' ';
        }
        file << "\n";
    }
}

void save_result(const vector<vector<int>>& result, int num_threads, int size) {
    fs::path dir = fs::path("data") / ("threads_" + to_string(num_threads));
    save_matrix(result, dir, "multiplied_" + to_string(size));
}

void save_report(const string& report) {
    fs::path dir = "data";
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    ofstream file(dir / "results.txt", ios::app);
    if (!file) {
        throw runtime_error("Can't open report file for writing");
    }
    file << report << "\n";
}

vector<vector<int>> load_matrix(int size, const string& filename) {
    fs::path filepath = fs::path("data") / to_string(size) / (filename + ".txt");
    ifstream file(filepath);
    if (!file) throw runtime_error("Cannot open file: " + filepath.string());

    vector<vector<int>> mat(size, vector<int>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            file >> mat[i][j];

    return mat;
}

__global__ void matrixMultiplyKernel(int* a, int* b, int* c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

vector<vector<int>> multiplyMatricesCUDA(const vector<vector<int>>& a, 
                                      const vector<vector<int>>& b, 
                                      int threadsPerBlock) {
    int size = a.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

    int* h_a = new int[size * size];
    int* h_b = new int[size * size];
    int* h_c = new int[size * size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_a[i * size + j] = a[i][j];
            h_b[i * size + j] = b[i][j];
        }
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size * size * sizeof(int));
    cudaMalloc(&d_b, size * size * sizeof(int));
    cudaMalloc(&d_c, size * size * sizeof(int));

    cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGrid((size + threadsPerBlock - 1) / threadsPerBlock, 
                      (size + threadsPerBlock - 1) / threadsPerBlock);

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlockDim>>>(d_a, d_b, d_c, size);

    cudaMemcpy(h_c, d_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = h_c[i * size + j];
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return result;
}

int main() {
    srand(time(0));
    bool use_existing = false;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "ERROR: No CUDA devices found" << endl;
        return 1;
    }
    cout << "Found CUDA devices: " << deviceCount << endl;

    vector<int> sizes = {100, 200, 300, 400, 500, 600};
    vector<int> thread_counts = {2, 4, 8, 12, 16};

    for (int size : sizes) {
        vector<vector<int>> A(size, vector<int>(size));
        vector<vector<int>> B(size, vector<int>(size));
        
        if (use_existing) {
            A = load_matrix(size, "A");
            B = load_matrix(size, "B");
        }
        else {
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    A[i][j] = XorShift128();
                    B[i][j] = XorShift128();
                }
            }
            save_matrix(A, fs::path("data") / to_string(size), "A");
            save_matrix(B, fs::path("data") / to_string(size), "B");
        }

        cout << "\nProcessing matrices " << size << "x" << size << "..." << endl;

        for (int threads : thread_counts) {
            cout << "Threads: " << threads << endl;
            
            auto start = chrono::high_resolution_clock::now();
            auto result = multiplyMatricesCUDA(A, B, threads);
            auto end = chrono::high_resolution_clock::now();

            save_result(result, threads, size);

            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            string report = "Size: " + to_string(size) + 
                          ", Threads: " + to_string(threads) + 
                          ", Time: " + to_string(duration.count()) + " ms";
            save_report(report);
            
            cout << report << endl;
        }
    }

    return 0;
}