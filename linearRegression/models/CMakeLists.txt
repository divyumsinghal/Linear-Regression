# CMakeList.txt : CMake project for linearRegression, include source and define
# project specific logic here.
#

cmake_minimum_required(VERSION 3.18)
project(LinearRegressionCUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (optimized for RTX 4050 - compute capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

# add libary for utils
add_subdirectory(utils)

# Create executable
add_library(LinearRegressionCUDA linearRegression.cu linearRegression.cuh)

# Set target properties
set_target_properties(LinearRegressionCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(LinearRegressionCUDA 
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Include directories
target_include_directories(LinearRegressionCUDA PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# CUDA compiler options
target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
>)

# Debug and Release configurations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
    target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
else()
    target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_options(LinearRegressionCUDA PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# Print some information
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")