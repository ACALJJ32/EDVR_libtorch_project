list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")
find_package(Torch REQUIRED) 

add_library(models edvr.cpp)

target_link_libraries(models ${TORCH_LIBRARIES})
target_include_directories(models PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")