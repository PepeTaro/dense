#include "utils/losses.h"
#include "common_utils.h"

#include <cmath>

#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dense;

TEST(TestMse,DoesEqual){
  Eigen::Tensor<double,1> predicts(3);
  Eigen::Tensor<double,1> labels(3);
  double expectation;
  
  predicts.setValues({0,0,0});
  labels.setValues({0,0,0});
  expectation = 0.0;
  EXPECT_NEAR(Mse(predicts,labels),expectation,abs_error);

  predicts.setValues({3,141592,-653589});
  labels.setValues({3,141592,-653589});
  expectation = 0.0;
  EXPECT_NEAR(Mse(predicts,labels),expectation,abs_error);

  predicts.setValues({1,2,3});
  labels.setValues({1,1,1});
  expectation = 2.5;
  EXPECT_NEAR(Mse(predicts,labels),expectation,abs_error);

  predicts.setValues({-2,5,-8});
  labels.setValues({3,-4,-1});
  expectation = 77.5;
  EXPECT_NEAR(Mse(predicts,labels),expectation,abs_error);
}


TEST(TestCategoricalCrossEntropy,DoesEqual){
  Eigen::Tensor<double,1> predicts(3);
  Eigen::Tensor<double,1> labels(3);
  double expectation;
  double factor;

  factor = 3*exp(0);
  expectation = log(3);
  predicts.setValues({0,0,0});
  labels.setValues({1,0,0});
  EXPECT_NEAR(CategoricalCrossEntropy(predicts,labels),expectation,abs_error);

  factor = exp(1)+2*exp(0);
  expectation = -log(exp(1)/factor);
  predicts.setValues({1,0,0});
  labels.setValues({1,0,0});
  EXPECT_NEAR(CategoricalCrossEntropy(predicts,labels),expectation,abs_error);

  factor = exp(1)+exp(-1)+exp(0);
  expectation = -log(exp(-1)/factor);
  predicts.setValues({1,-1,0});
  labels.setValues({0,1,0});
  EXPECT_NEAR(CategoricalCrossEntropy(predicts,labels),expectation,abs_error);

  factor = exp(3)+exp(-4)+exp(0);
  expectation = -log(exp(0)/factor);
  predicts.setValues({3,-4,0});
  labels.setValues({0,0,1});
  EXPECT_NEAR(CategoricalCrossEntropy(predicts,labels),expectation,abs_error);
}

TEST(TestGradientOfMse,DoesEqual){
  Eigen::Tensor<double,1> predicts(3);
  Eigen::Tensor<double,1> labels(3);
  Eigen::Tensor<double,1> expectation(3);
  
  predicts.setValues({0,0,0});
  labels.setValues({1,0,0});
  expectation.setValues({-1,0,0});
  EXPECT_NEAR_RANK1_TENSOR(GradientOfMse(predicts,labels),expectation);

  predicts.setValues({1,-1,0});
  labels.setValues({1,-1,0});
  expectation.setValues({0,0,0});
  EXPECT_NEAR_RANK1_TENSOR(GradientOfMse(predicts,labels),expectation);

  predicts.setValues({1,-1,0});
  labels.setValues({-1,1,0});
  expectation.setValues({2,-2,0});
  EXPECT_NEAR_RANK1_TENSOR(GradientOfMse(predicts,labels),expectation);
}

TEST(TestGradientOfCategoricalCrossEntropy,DoesEqual){
  Eigen::Tensor<double,1> predicts(3);
  Eigen::Tensor<double,1> labels(3);
  Eigen::Tensor<double,1> expectation(3);
  
  predicts.setValues({0,0,0});
  labels.setValues({1,0,0});
  expectation.setValues({-2.0/3,1.0/3,1.0/3});
  EXPECT_NEAR_RANK1_TENSOR(GradientOfCategoricalCrossEntropy(predicts,labels),expectation);
}
