#include "models/linearRegression.h"
#include <iostream>
#include <iomanip>
#include <vector>

using data_type = double;

int main() {
    std::cout << "=== CUDA Linear Regression Calculator ===\n\n";
    
    // Example: small dataset
    int n_samples = 5, n_features = 2;
    std::cout << "Dataset: " << n_samples << " samples, " << n_features << " features\n\n";

    // Allocate host data
    std::vector<data_type> h_X_flat(n_samples * n_features);
    std::vector<data_type> h_y(n_samples);

    // Fill with simple linear relationship: y = 2*x1 + 3*x2 + 1 + noise
    std::cout << "Generating synthetic data (y = 2*x1 + 3*x2 + 1):\n";
    for (int i = 0; i < n_samples; ++i) {
        data_type x1 = i + 1.0;           // x1: 1, 2, 3, 4, 5
        data_type x2 = (i + 1) * 0.5;     // x2: 0.5, 1.0, 1.5, 2.0, 2.5
        
        // Store in column-major format: [x1_1, x1_2, ..., x1_n, x2_1, x2_2, ..., x2_n]
        h_X_flat[0 * n_samples + i] = x1;  // First feature column
        h_X_flat[1 * n_samples + i] = x2;  // Second feature column
        
        h_y[i] = 2.0 * x1 + 3.0 * x2 + 1.0;  // True relationship
        
        std::cout << "Sample " << i+1 << ": x1=" << x1 
                  << ", x2=" << x2 << ", y=" << h_y[i] << "\n";
    }
    std::cout << "\n";

    // Create linear regression model and fit
    LinearRegression<data_type> lr;
    std::vector<data_type> weights = lr.fit(h_X_flat, h_y, n_samples, n_features);

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