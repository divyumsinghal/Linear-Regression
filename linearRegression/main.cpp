#include <models/linearRegression.cuh>

using data_type = double;

int main() {
    // Example: small dataset
    int n_samples = 100, n_features = 3;

    // Allocate host data
    thrust::host_vector<thrust::host_vector<data_type>> h_X(n_features , thrust::host_vector<data_type>(n_samples));
    thrust::host_vector<data_type> h_y(n_samples);

    // Fill h_X, h_y with dummy data

    for (int i = 0; i < n_samples; ++i) {
        h_y[i] = static_cast<data_type>(i);
        for (int j = 0; j < n_features; ++j) {
            h_X[j][i] = static_cast<data_type>(i + j);
        }
    }

    // Copy to device
    thrust::device_vector<thrust::device_vector<data_type>> d_vec_x = h_X;
    thrust::device_vector<data_type> d_vec_y = h_y;

    // Allocate device memory
    // data_type** d_X = thrust::raw_pointer_cast(d_vec_x.data());
    // data_type* d_y = thrust::raw_pointer_cast(d_vec_y.data());

    LinearRegression<data_type> lr;
    thrust::host_vector<data_type> w = lr.fit(d_vec_x, d_vec_y);


    return 0;
}
