cmake_minimum_required(VERSION 3.10)

project(examples)

# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(INCLUDE_DIR ../include)
SET(SRC_DIR ../src)
SET(DENSE_LIB_SRC_FILES ${SRC_DIR}/dense.cpp ${SRC_DIR}/utils/activations.cpp ${SRC_DIR}/utils/losses.cpp ${SRC_DIR}/utils/utils.cpp)
SET(GCC_COVERAGE_COMPILE_FLAGS "-O3 -fopenmp -g")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_library(dense ${DENSE_LIB_SRC_FILES})
target_include_directories(dense PUBLIC ${INCLUDE_DIR})
target_link_libraries(dense Eigen3::Eigen)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist LINK_PUBLIC dense Eigen3::Eigen)
