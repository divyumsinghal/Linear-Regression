#include "matrix_utils.cuh"

// CUDA kernel to add bias column
template<typename T>
__global__ void add_bias_kernel(T* d_X_with_bias, const T* d_X, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < rows && idy < cols + 1) {
        if (idy == 0) {
            // First column is bias (ones)
            d_X_with_bias[idy * rows + idx] = 1.0;
        } else {
            // Copy original data
            d_X_with_bias[idy * rows + idx] = d_X[(idy - 1) * rows + idx];
        }
    }
}

template<typename T>
void add_bias_column(T* d_X_with_bias, const T* d_X, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x, (cols + 1 + block.y - 1) / block.y);
    
    add_bias_kernel<<<grid, block>>>(d_X_with_bias, d_X, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void print_device_matrix(const T* d_matrix, int rows, int cols, const char* name) {
    T* h_matrix = new T[rows * cols];
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));
    
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_matrix[j * rows + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    delete[] h_matrix;
}

template<typename T>
void print_device_vector(const T* d_vector, int size, const char* name) {
    T* h_vector = new T[size];
    CUDA_CHECK(cudaMemcpy(h_vector, d_vector, size * sizeof(T), cudaMemcpyDeviceToHost));
    
    std::cout << name << " (" << size << "):\n";
    for (int i = 0; i < size; i++) {
        std::cout << h_vector[i] << " ";
    }
    std::cout << "\n\n";
    
    delete[] h_vector;
}

// Explicit template instantiations
template void add_bias_column<float>(float*, const float*, int, int);
template void add_bias_column<double>(double*, const double*, int, int);
template void print_device_matrix<float>(const float*, int, int, const char*);
template void print_device_matrix<double>(const double*, int, int, const char*);
template void print_device_vector<float>(const float*, int, const char*);
template void print_device_vector<double>(const double*, int, const char*);