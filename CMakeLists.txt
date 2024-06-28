cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")

# set nvcc path and architecture
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 86)

# set project language
project(coh LANGUAGES CUDA CXX)

# Set the C++ and CUDA standards and ensure they are required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") # Added -O3 optimization for C++

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3") # Added -O3 optimization for CUDA./

# Find OpenMP package
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3") # Added -O3 for C with OpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3") # Added -O3 for C++ with OpenMP
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Include directories
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Fetch additional dependencies
include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

# Add source file list
add_subdirectory(src)

# Add executable
add_executable(coh main.cu)

# Link source files with required libraries
target_link_libraries(coh src argparse)

# Optional: Specify target properties for CUDA
set_target_properties(coh PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optional: Use ccache for faster recompilation if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()
