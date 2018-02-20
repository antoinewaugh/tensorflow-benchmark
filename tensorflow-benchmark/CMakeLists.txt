cmake_minimum_required(VERSION 3.5)
project(tensorflow-benchmark)

set (CMAKE_CXX_STANDARD 11)

find_package(TensorflowCC COMPONENTS Shared)

add_executable(tensorflow-benchmark main.cpp)
target_link_libraries(tensorflow-benchmark TensorflowCC::Shared)

