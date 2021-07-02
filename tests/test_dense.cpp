#include "dense.h"
#include "gtest/gtest.h"

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dense;

const double abs_error = 0.00001;

TEST(TestDenseParameters,DoesEqual){
  Dense dense1({784,64,32,10},{Activations::Sigmoid,Activations::Sigmoid,Activations::Sigmoid},Losses::Mse);
  dense1.SetLearningRate(0.0);
  dense1.SetWeightDecay(0.0);  
  EXPECT_EQ(dense1.GetNumOfLayers(),3);
  EXPECT_EQ(dense1.GetLearningRate(),0);
  EXPECT_EQ(dense1.GetWeightDecay(),0);

  Dense dense2({32,10},{Activations::Sigmoid},Losses::Mse);
  dense2.SetLearningRate(0.001);
  dense2.SetWeightDecay(0.1);  
  EXPECT_EQ(dense2.GetNumOfLayers(),1);
  EXPECT_EQ(dense2.GetLearningRate(),0.001);
  EXPECT_EQ(dense2.GetWeightDecay(),0.1);  
}
