list(APPEND CMAKE_PREFIX_PATH "/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/code/libtorch_learning/EDVRProject/libtorch")
find_package(Torch REQUIRED) 

add_library(dcn edvr.cpp)

target_link_libraries(dcn ${TORCH_LIBRARIES})
target_include_directories(dcn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")