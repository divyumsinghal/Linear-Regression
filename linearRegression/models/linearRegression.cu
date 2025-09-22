#include "linearRegression.cuh"

template <typename data_type>
LinearRegression<data_type>::LinearRegression() : d_weights(nullptr), n_features(0) {}

template <typename data_type>
LinearRegression<data_type>::~LinearRegression() {
    if (d_weights != nullptr) {
        cudaFree(d_weights);
    }
}

template <typename data_type>
thrust::host_vector<data_type> LinearRegression<data_type>::fit(
    const thrust::device_vector<thrust::device_vector<data_type>>& d_vec_x, 
    const thrust::device_vector<data_type>& d_vec_y)
{
    std::cout << "Starting Linear Regression fit...\n";
    
    // Calculate weights using normal equation: weights = (X^T * X)^-1 * X^T * y
    thrust::host_vector<data_type> weights = calculate_linear_regression_weights<data_type>(d_vec_x, d_vec_y);
    
    std::cout << "Linear Regression fit completed!\n";
    return weights;
}

// Explicit template instantiations
template class LinearRegression<float>;
template class LinearRegression<double>;