cmake_minimum_required(VERSION 3.10)
 
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

add_subdirectory(benchmark)
include_directories(benchmark/include)

find_package(Threads REQUIRED)
find_package(TensorflowCC COMPONENTS Shared)

set(SOURCE_FILES
    graph_runner.h
    bench.cpp
)

add_executable(tensorflow-benchmark ${SOURCE_FILES})

target_link_libraries(tensorflow-benchmark
        benchmark
        ${CMAKE_THREAD_LIBS_INIT}
        TensorflowCC::Shared
)
