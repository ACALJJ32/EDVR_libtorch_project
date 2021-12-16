set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

list(APPEND EXTRA_INCLUDES "/usr/include/python")
list(APPEND CMAKE_PREFIX_PATH "/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/code/libtorch_learning/EDVRProject/libtorch")

find_package(CUDA REQUIRED)
find_package(Python REQUIRED)
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

add_library(
    dcn SHARED
    src/deform_conv_ext.cpp
    src/deform_conv_cuda.cpp
    src/deform_conv_cuda_kernel.cu
)

target_link_libraries(dcn ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(dcn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${EXTRA_INCLUDES})

target_compile_definitions(dcn PUBLIC DCN_VERSION=1)