// linearRegression.cpp : Defines the entry point for the application.
//

#include "models/linear_regression.cuh"
#include "linearRegression.cuh"


template <typename data_type>
inline LinearRegression<data_type>::LinearRegression() {}

template <typename data_type>
inline LinearRegression<data_type>::~LinearRegression() {}

template <typename data_type>
thrust::host_vector<data_type> LinearRegression<data_type>::fit(const thrust::device_vector<thrust::device_vector<data_type>> d_vec_x, thrust::device_vector<data_type> d_vec_y)
{
    // Add 1s to the X matrix for the constant term


    // Calculate weights using normal equation: weights = (X^T * X)^-1 * X^T * y
    thrust::host_vector<data_type> weights = calculate_linear_multiplication<data_type><<< >>>(d_vec_x, d_vec_y);

    return weights;
}
