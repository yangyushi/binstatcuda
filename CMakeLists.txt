cmake_minimum_required(VERSION 3.20)

project(
    binstatcuda
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(
        CMAKE_CUDA_ARCHITECTURES
        native
        CACHE STRING "CUDA architectures to compile for."
    )
endif()

# bug2025, https://github.com/pybind/pybind11/issues/4825
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

find_package(Python3 COMPONENTS Development Interpreter REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(
    binstatcuda_core
    src/cpp/bindings.cpp
    src/cuda/device_info.cu
    src/cuda/histogram.cu
)

target_include_directories(
    binstatcuda_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    binstatcuda_core
    PRIVATE
        pybind11::module
        CUDA::cudart
)

target_compile_features(binstatcuda_core PRIVATE cxx_std_17)

set_target_properties(
    binstatcuda_core
    PROPERTIES
        OUTPUT_NAME "_core"
        PREFIX ""
        CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
    target_compile_options(binstatcuda_core PRIVATE /W4)
else()
    target_compile_options(binstatcuda_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(
    TARGETS binstatcuda_core
    LIBRARY DESTINATION binstatcuda
)
