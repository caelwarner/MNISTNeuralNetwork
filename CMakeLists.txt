cmake_minimum_required(VERSION 3.28.3)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

if(MSVC)
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /Zi")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT} /DEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Oy-")
endif()

if(NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
endif()

add_executable(NeuralNetwork src/main.c
        src/linalg.c
        src/linalg.h
        src/io.c
        src/io.h
        src/main.h)


if(NOT MSVC)
    target_link_libraries(NeuralNetwork m)
endif()
    