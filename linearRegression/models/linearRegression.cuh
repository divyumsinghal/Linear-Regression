#pragma once

#include "utils/calculateMatrixResult.cuh"
#include "utils/matrix_utils.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename data_type>
class LinearRegression {
private:
    data_type* d_weights;  // GPU weights
    int n_features;

public:
    LinearRegression();
    ~LinearRegression();

    thrust::host_vector<data_type> fit(const thrust::device_vector<thrust::device_vector<data_type>>& d_vec_x, 
                                      const thrust::device_vector<data_type>& d_vec_y);
};