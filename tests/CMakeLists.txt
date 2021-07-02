cmake_minimum_required(VERSION 3.10)

project(tests)

# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Gtest
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

SET(INCLUDE_DIR ../include)
SET(SRC_DIR ../src)
SET(DENSE_LIB_SRC_FILES ${SRC_DIR}/dense.cpp ${SRC_DIR}/utils/activations.cpp ${SRC_DIR}/utils/losses.cpp ${SRC_DIR}/utils/utils.cpp)
SET(GCC_COVERAGE_COMPILE_FLAGS "-O3 -fopenmp -g")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_library(dense ${DENSE_LIB_SRC_FILES})
target_include_directories(dense PUBLIC ${INCLUDE_DIR})
target_link_libraries(dense Eigen3::Eigen)

add_executable(test_all test_all.cpp test_activations.cpp test_losses.cpp test_dense.cpp common_utils.cpp)
target_link_libraries(test_all LINK_PUBLIC dense ${GTEST_BOTH_LIBRARIES})

add_executable(test_activations test_activations.cpp common_utils.cpp)
target_link_libraries(test_activations LINK_PUBLIC dense ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen)

add_executable(test_losses test_losses.cpp common_utils.cpp)
target_link_libraries(test_losses LINK_PUBLIC dense ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen)

add_executable(test_dense test_dense.cpp common_utils.cpp)
target_link_libraries(test_dense LINK_PUBLIC dense ${GTEST_BOTH_LIBRARIES} Eigen3::Eigen)
