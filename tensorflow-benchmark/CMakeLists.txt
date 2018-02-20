cmake_minimum_required(VERSION 3.5)
project(tensorflow-runner)

set (CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)
find_package(TensorflowCC COMPONENTS Shared)

add_subdirectory(bench)

add_executable(tensorflow-runner main.cpp)
target_link_libraries(tensorflow-runner TensorflowCC::Shared)

