list(APPEND CMAKE_PREFIX_PATH "/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/code/libtorch_learning/EDVRProject/libtorch")
list(APPEND EXTRA_INCLUDES "/usr/include/python")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

find_package(Torch REQUIRED) 
find_package(Python REQUIRED)

add_library(models edvr.cpp)
add_subdirectory(dcn)

target_link_libraries(models PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} dcn)
target_include_directories(models PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/dcn" "${CMAKE_CURRENT_SOURCE_DIR}")