#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <iostream>

template<typename T>
void print_device_matrix(const T* d_matrix, int rows, int cols, const char* name);

template<typename T>
void print_device_vector(const T* d_vector, int size, const char* name);

template<typename T>
void convert_thrust_to_matrix(const thrust::device_vector<thrust::device_vector<T>>& thrust_matrix, 
                             T** d_matrix, int& rows, int& cols);

template<typename T>
void add_bias_column(T* d_X_with_bias, const T* d_X, int rows, int cols);