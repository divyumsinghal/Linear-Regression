#pragma once

#include <vector>

template<typename data_type>
class LinearRegression {
private:
    data_type* d_weights;  // GPU weights
    int n_features;

public:
    LinearRegression();
    ~LinearRegression();

    std::vector<data_type> fit(const std::vector<data_type>& h_X_flat, 
                              const std::vector<data_type>& h_vec_y,
                              int n_samples,
                              int n_features);
};