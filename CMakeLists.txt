cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(evol_cpp)

find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_executable(evol evol.cpp)
target_link_libraries(evol "${TORCH_LIBRARIES}")
set_property(TARGET evol PROPERTY CXX_STANDARD 17)
