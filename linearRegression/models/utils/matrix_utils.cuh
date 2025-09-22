#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <iostream>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// CUSOLVER API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            std::printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                 \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)
    
template<typename T>
void print_device_matrix(const T* d_matrix, int rows, int cols, const char* name);

template<typename T>
void print_device_vector(const T* d_vector, int size, const char* name);

template<typename T>
void convert_thrust_to_matrix(const thrust::device_vector<thrust::device_vector<T>>& thrust_matrix, 
                             T** d_matrix, int& rows, int& cols);

template<typename T>
void add_bias_column(T* d_X_with_bias, const T* d_X, int rows, int cols);