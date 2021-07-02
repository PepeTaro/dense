#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "gtest/gtest.h"
#include <unsupported/Eigen/CXX11/Tensor>

const double abs_error = 0.00001;

void EXPECT_NEAR_RANK1_TENSOR(const Eigen::Tensor<double,1>& test_tensor,const Eigen::Tensor<double,1>& expected_tensor);

#endif// TEST_UTILS_H
