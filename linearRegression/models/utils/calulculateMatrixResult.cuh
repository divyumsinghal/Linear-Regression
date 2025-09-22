#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T>
thrust::host_vector<T> calculate_linear_regression_weights(
    const thrust::device_vector<thrust::device_vector<T>>& d_vec_x,
    const thrust::device_vector<T>& d_vec_y);