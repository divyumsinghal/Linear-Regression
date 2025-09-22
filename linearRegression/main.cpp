#include "models/linearRegression.cuh"
#include <iostream>
#include <iomanip>

using data_type = double;

int main() {
    std::cout << "=== CUDA Linear Regression Calculator ===\n\n";
    
    // Example: small dataset
    int n_samples = 5, n_features = 2;
    std::cout << "Dataset: " << n_samples << " samples, " << n_features << " features\n\n";

    // Allocate host data
    thrust::host_vector<thrust::host_vector<data_type>> h_X(n_features, thrust::host_vector<data_type>(n_samples));
    thrust::host_vector<data_type> h_y(n_samples);

    // Fill with simple linear relationship: y = 2*x1 + 3*x2 + 1 + noise
    std::cout << "Generating synthetic data (y = 2*x1 + 3*x2 + 1):\n";
    for (int i = 0; i < n_samples; ++i) {
        h_X[0][i] = i + 1.0;           // x1: 1, 2, 3, 4, 5
        h_X[1][i] = (i + 1) * 0.5;     // x2: 0.5, 1.0, 1.5, 2.0, 2.5
        h_y[i] = 2.0 * h_X[0][i] + 3.0 * h_X[1][i] + 1.0;  // True relationship
        
        std::cout << "Sample " << i+1 << ": x1=" << h_X[0][i] 
                  << ", x2=" << h_X[1][i] << ", y=" << h_y[i] << "\n";
    }
    std::cout << "\n";

    // Copy to device
    thrust::device_vector<thrust::device_vector<data_type>> d_vec_x(n_features);
    for (int j = 0; j < n_features; ++j) {
        d_vec_x[j] = h_X[j];
    }
    thrust::device_vector<data_type> d_vec_y = h_y;

    // Create linear regression model and fit
    LinearRegression<data_type> lr;
    thrust::host_vector<data_type> weights = lr.fit(d_vec_x, d_vec_y);

    // Print results
    std::cout << "=== FINAL RESULTS ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Calculated weights:\n";
    std::cout << "  Bias (intercept): " << weights[0] << "\n";
    for (int i = 1; i < weights.size(); ++i) {
        std::cout << "  Feature " << i << " weight: " << weights[i] << "\n";
    }
    
    std::cout << "\nExpected weights:\n";
    std::cout << "  Bias (intercept): 1.000000\n";
    std::cout << "  Feature 1 weight: 2.000000\n";
    std::cout << "  Feature 2 weight: 3.000000\n";

    return 0;
}