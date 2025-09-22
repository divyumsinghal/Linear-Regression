cmake_minimum_required(VERSION 3.18)
project(Main LANGUAGES CXX CUDA)

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

# Add subdirectory for models
add_subdirectory(models)

# Create executable
add_executable(Main main.cpp)

# Set target properties
set_target_properties(Main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(Main 
    LinearRegressionCUDA
    Utils
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Include directories
target_include_directories(Main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(Main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# CUDA compiler options
target_compile_options(Main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    --expt-relaxed-constexpr
>)

# Debug and Release configurations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(Main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
    target_compile_options(Main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
else()
    target_compile_options(Main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_options(Main PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# Print some information
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")