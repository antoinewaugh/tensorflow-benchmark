cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)

project(tensorflow-runner)

include_directories(.)
add_subdirectory(bench)
add_subdirectory(src)


