cmake_minimum_required(VERSION 3.18)
project(LinearRegressionCUDA LANGUAGES CXX CUDA)

# Add subdirectory for utils
add_subdirectory(utils)

# Create library
add_library(LinearRegressionCUDA linearRegression.cu)

# Set target properties
set_target_properties(LinearRegressionCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(LinearRegressionCUDA 
    Utils
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Include directories
target_include_directories(LinearRegressionCUDA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# CUDA compiler options
target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
>)