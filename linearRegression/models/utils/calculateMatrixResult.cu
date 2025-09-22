#include "calculateMatrixResult.cuh"
#include "matrix_utils.cuh"
#include <memory>

template<typename T>
thrust::host_vector<T> calculate_linear_regression_weights(
    const thrust::device_vector<T>& d_X_flat,
    const thrust::device_vector<T>& d_vec_y,
    int n_samples,
    int n_features)
{
    int n_features_with_bias = n_features + 1;

    std::cout << "Input dimensions: " << n_samples << " samples, " << n_features << " features\n";

    // Get raw pointers from thrust vectors
    const T* d_X_raw = thrust::raw_pointer_cast(d_X_flat.data());
    const T* d_y = thrust::raw_pointer_cast(d_vec_y.data());

    // Create X matrix with bias column
    T* d_X_with_bias;
    CUDA_CHECK(cudaMalloc(&d_X_with_bias, n_samples * n_features_with_bias * sizeof(T)));
    
    // Add bias column (column of ones)
    add_bias_column(d_X_with_bias, d_X_raw, n_samples, n_features);

    // Print input data
    std::cout << "\nInput Matrix X (with bias):\n";
    print_device_matrix(d_X_with_bias, n_samples, n_features_with_bias, "X");
    
    std::cout << "\nInput Vector y:\n";
    print_device_vector(d_y, n_samples, "y");

    // Initialize cuBLAS and cuSOLVER
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // Allocate memory for intermediate calculations
    T* d_XtX;  // X^T * X
    T* d_Xty;  // X^T * y
    T* d_weights; // Final weights
    
    CUDA_CHECK(cudaMalloc(&d_XtX, n_features_with_bias * n_features_with_bias * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_Xty, n_features_with_bias * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_weights, n_features_with_bias * sizeof(T)));

    const T alpha = 1.0;
    const T beta = 0.0;

    // Calculate X^T * X using cuBLAS
    // Note: cuBLAS uses column-major format, so we need to be careful with dimensions
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                n_features_with_bias, n_features_with_bias, n_samples,
                                &alpha, d_X_with_bias, n_samples,
                                d_X_with_bias, n_samples,
                                &beta, d_XtX, n_features_with_bias));
    } else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                n_features_with_bias, n_features_with_bias, n_samples,
                                &alpha, d_X_with_bias, n_samples,
                                d_X_with_bias, n_samples,
                                &beta, d_XtX, n_features_with_bias));
    }

    // Add tiny regularization to diagonal to handle numerical precision issues
    const T regularization = 1e-14;
    T* h_XtX = new T[n_features_with_bias * n_features_with_bias];
    CUDA_CHECK(cudaMemcpy(h_XtX, d_XtX, n_features_with_bias * n_features_with_bias * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_features_with_bias; i++) {
        h_XtX[i * n_features_with_bias + i] += regularization;
    }
    CUDA_CHECK(cudaMemcpy(d_XtX, h_XtX, n_features_with_bias * n_features_with_bias * sizeof(T), cudaMemcpyHostToDevice));
    delete[] h_XtX;

    // Calculate X^T * y using cuBLAS
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T,
                                n_samples, n_features_with_bias,
                                &alpha, d_X_with_bias, n_samples,
                                d_y, 1,
                                &beta, d_Xty, 1));
    } else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_T,
                                n_samples, n_features_with_bias,
                                &alpha, d_X_with_bias, n_samples,
                                d_y, 1,
                                &beta, d_Xty, 1));
    }

    // Solve the linear system (X^T * X) * weights = X^T * y using cuSOLVER
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Query workspace size
    int lwork = 0;
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolver_handle, 
                                                  n_features_with_bias, n_features_with_bias,
                                                  d_XtX, n_features_with_bias, &lwork));
    } else if constexpr (std::is_same_v<T, double>) {
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolver_handle, 
                                                  n_features_with_bias, n_features_with_bias,
                                                  d_XtX, n_features_with_bias, &lwork));
    }

    T* d_workspace;
    int* d_pivot;
    CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_pivot, n_features_with_bias * sizeof(int)));

    // LU factorization
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSgetrf(cusolver_handle,
                                       n_features_with_bias, n_features_with_bias,
                                       d_XtX, n_features_with_bias,
                                       d_workspace, d_pivot, d_info));
    } else if constexpr (std::is_same_v<T, double>) {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolver_handle,
                                       n_features_with_bias, n_features_with_bias,
                                       d_XtX, n_features_with_bias,
                                       d_workspace, d_pivot, d_info));
    }

    // Check for errors in LU factorization
    int info_h;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        std::cerr << "LU factorization failed with info = " << info_h << std::endl;
        throw std::runtime_error("LU factorization failed");
    }

    // Copy X^T * y to weights vector (will be overwritten with solution)
    CUDA_CHECK(cudaMemcpy(d_weights, d_Xty, n_features_with_bias * sizeof(T), cudaMemcpyDeviceToDevice));

    // Solve the system
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N,
                                       n_features_with_bias, 1,
                                       d_XtX, n_features_with_bias,
                                       d_pivot, d_weights, n_features_with_bias, d_info));
    } else if constexpr (std::is_same_v<T, double>) {
        CUSOLVER_CHECK(cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N,
                                       n_features_with_bias, 1,
                                       d_XtX, n_features_with_bias,
                                       d_pivot, d_weights, n_features_with_bias, d_info));
    }

    // Check for errors in solving
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        std::cerr << "Linear system solve failed with info = " << info_h << std::endl;
        throw std::runtime_error("Linear system solve failed");
    }

    // Print calculated weights
    std::cout << "\nCalculated weights:\n";
    print_device_vector(d_weights, n_features_with_bias, "weights");

    // Copy result to host
    thrust::host_vector<T> h_weights(n_features_with_bias);
    CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, n_features_with_bias * sizeof(T), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_X_with_bias));
    CUDA_CHECK(cudaFree(d_XtX));
    CUDA_CHECK(cudaFree(d_Xty));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_pivot));
    CUDA_CHECK(cudaFree(d_info));

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

    return h_weights;
}

// Explicit template instantiations
template thrust::host_vector<float> calculate_linear_regression_weights(
    const thrust::device_vector<float>&, 
    const thrust::device_vector<float>&,
    int, int);

template thrust::host_vector<double> calculate_linear_regression_weights(
    const thrust::device_vector<double>&, 
    const thrust::device_vector<double>&,
    int, int);