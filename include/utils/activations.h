#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils/utils.h"

namespace dense{
  enum class Activations {
    Relu,
    Sigmoid,
    Null// ダミー用。
  };

  typedef double (*activation_type)(double);

  activation_type GetDerivativeActivation(Activations activation);
  activation_type GetActivation(Activations activation);

  double Relu(double x);
  double Sigmoid(double z);
  double DerivativeOfSigmoid(double z);
  double DerivativeOfRelu(double x);
  double LogSumExp(const Eigen::Tensor<double,1>& logits);
  Eigen::Tensor<double,1> Softmax(const Eigen::Tensor<double,1>& logits);

};//namespace dense
#endif// ACTIVATIONS_H
