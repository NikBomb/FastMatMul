cmake_minimum_required(VERSION 3.15)
project(FastMatMult LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FASTMATMULT_ENABLE_MARCH_NATIVE "Enable -march=native for extra performance" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O3 -DNDEBUG)
    if(FASTMATMULT_ENABLE_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
elseif(MSVC)
    add_compile_options(/O2 /DNDEBUG)
endif()

add_executable(fast_mat_mult main.cpp)
