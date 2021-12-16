cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(main)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_LIBRARY_PATH "/usr/local/cuda/targets/x86_64-linux/lib")
set(OpenCV_DIR /media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/code/libtorch_learning/EDVRProject/opencv)


list(APPEND CMAKE_PREFIX_PATH "/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/code/libtorch_learning/EDVRProject/libtorch")
list(APPEND EXTRA_INCLUDES "/usr/include/python")

find_package(CUDA REQUIRED)
find_package(Python REQUIRED)
find_package(Torch REQUIRED)  
find_package(OpenCV REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(models)
add_executable(main main.cpp)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${TORCH_PYTHON_LIBRARY})
target_include_directories(main PUBLIC "${PROHECT_BINARY_DIR}" ${EXTRA_INCLUDES})

set_property(TARGET main PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

