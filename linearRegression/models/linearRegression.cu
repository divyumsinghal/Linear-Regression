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
std::vector<data_type> LinearRegression<data_type>::fit(
    const std::vector<data_type>& h_X_flat, 
    const std::vector<data_type>& h_vec_y,
    int n_samples,
    int n_features)
{
    std::cout << "Starting Linear Regression fit...\n";
    
    // Convert std::vector to thrust::device_vector
    thrust::device_vector<data_type> d_X_flat(h_X_flat.begin(), h_X_flat.end());
    thrust::device_vector<data_type> d_vec_y(h_vec_y.begin(), h_vec_y.end());
    
    // Calculate weights using normal equation: weights = (X^T * X)^-1 * X^T * y
    thrust::host_vector<data_type> weights_thrust = calculate_linear_regression_weights<data_type>(
        d_X_flat, d_vec_y, n_samples, n_features);
    
    // Convert thrust::host_vector to std::vector
    std::vector<data_type> weights(weights_thrust.begin(), weights_thrust.end());
    
    std::cout << "Linear Regression fit completed!\n";
    return weights;
}

// Explicit template instantiations
template class LinearRegression<float>;
template class LinearRegression<double>;