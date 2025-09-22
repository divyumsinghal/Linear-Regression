# CMakeList.txt : CMake project for Utils library, include source and define
# project specific logic here.
#

cmake_minimum_required(VERSION 3.18)
project(Utils LANGUAGES CXX CUDA)

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

# Create Utils library with CUDA sources and headers
add_library(Utils 
    matrix_utils.cu 
    calculateMatrixResult.cu
    matrix_utils.cuh 
    calculateMatrixResult.cuh
)

# Set target properties for CUDA compilation
set_target_properties(Utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Link required CUDA libraries
target_link_libraries(Utils 
    PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Include directories - make them public so dependent targets can use them
target_include_directories(Utils 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler-specific options for C++
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(Utils PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
    )
endif()

# CUDA compiler options
target_compile_options(Utils PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        --maxrregcount=64
    >
)

# Debug and Release configurations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(Utils PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g -O0>
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
    )
    target_compile_definitions(Utils PRIVATE DEBUG=1)
else()
    target_compile_options(Utils PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
    )
endif()

# Enable verbose compilation for debugging
if(CMAKE_VERBOSE_MAKEFILE)
    target_compile_options(Utils PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--verbose>
    )
endif()

# Print configuration information
message(STATUS "Utils Library Configuration:")
message(STATUS "  CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")