#include "utils/activations.h"
#include "common_utils.h"

#include <cmath>

#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dense;

TEST(TestRelu,DoesEqual){
  EXPECT_NEAR(Relu(0.0),0.0,abs_error);
  EXPECT_NEAR(Relu(1.0),1.0,abs_error);
  EXPECT_NEAR(Relu(-1.0),0.0,abs_error);  
}

TEST(TestSigmoid,DoesEqual){
  EXPECT_NEAR(Sigmoid(0.0),0.5,abs_error);
  EXPECT_NEAR(Sigmoid(1.0),1/(1+exp(-1)),abs_error);
  EXPECT_NEAR(Sigmoid(-1.0),1/(1+exp(1)),abs_error);  
}

TEST(TestDerivativeOfRelu,DoesEqual){
  EXPECT_NEAR(DerivativeOfRelu(0.0),1.0,abs_error);
  EXPECT_NEAR(DerivativeOfRelu(1.0),1.0,abs_error);
  EXPECT_NEAR(DerivativeOfRelu(-1.0),0.0,abs_error);  
}

TEST(TestDerivativeOfSigmoid,DoesEqual){
  EXPECT_NEAR(DerivativeOfSigmoid(0.0),0.25,abs_error);
  EXPECT_NEAR(DerivativeOfSigmoid(1.0),1/(2+exp(1)+exp(-1)),abs_error);
  EXPECT_NEAR(DerivativeOfSigmoid(-1.0),1/(2+exp(1)+exp(-1)),abs_error);  
}

TEST(TestLogSumExp,DoesEqual){
  Eigen::Tensor<double,1> logits(3);
  
  logits.setValues({1,1,1});  
  EXPECT_NEAR(LogSumExp(logits),(log(3)+1),abs_error);

  logits.setValues({0,0,0});  
  EXPECT_NEAR(LogSumExp(logits),log(3),abs_error);

  logits.setValues({-1,0,-1});  
  EXPECT_NEAR(LogSumExp(logits),log(2/exp(1) + 1),abs_error);

  logits.setValues({-1,0,1});  
  EXPECT_NEAR(LogSumExp(logits),log(exp(1)+exp(-1) + 1),abs_error);

}

TEST(TestSoftmax,DoesEqual){
  Eigen::Tensor<double,1> logits(3);
  Eigen::Tensor<double,1> expectation(3);
  double factor;

  factor = 3*exp(1);
  logits.setValues({1,1,1});
  expectation.setValues({exp(1)/factor,exp(1)/factor,exp(1)/factor});
  EXPECT_NEAR_RANK1_TENSOR(Softmax(logits),expectation);

  factor = 3*exp(0);
  logits.setValues({0,0,0});
  expectation.setValues({exp(0)/factor,exp(0)/factor,exp(0)/factor});
  EXPECT_NEAR_RANK1_TENSOR(Softmax(logits),expectation);

  factor = exp(0)+2*exp(-1);
  logits.setValues({-1,0,-1});
  expectation.setValues({exp(-1)/factor,exp(0)/factor,exp(-1)/factor});
  EXPECT_NEAR_RANK1_TENSOR(Softmax(logits),expectation);
  
  factor = exp(-1)+exp(0)+exp(1);
  logits.setValues({-1,0,1});
  expectation.setValues({exp(-1)/factor,exp(0)/factor,exp(1)/factor});
  EXPECT_NEAR_RANK1_TENSOR(Softmax(logits),expectation);
}
