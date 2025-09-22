// linearRegression.cuh : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "utils/calculateMatrixResult.cu"
#include "utils/matrix_utils.cu"

template<typename data_type>
class LinearRegression {
private:
    data_type* d_weights;  // GPU weights
    int n_features;

public:
    LinearRegression();
    ~LinearRegression();

    thrust::host_vector<data_type> fit(const thrust::device_vector<thrust::device_vector<data_type>>& d_vec_x, const thrust::device_vector<data_type>& d_vec_y);
};
