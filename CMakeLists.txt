cmake_minimum_required(VERSION 3.28.3)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork main.c)
target_link_libraries(NeuralNetwork m)
