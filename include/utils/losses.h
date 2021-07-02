#ifndef LOSSES_H
#define LOSSES_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils/activations.h"
#include "utils/utils.h"

namespace dense{
  enum class Losses{
    Mse,
    CategoricalCrossEntropy,
    Null// ダミー用。
  };

  typedef Eigen::Tensor<double,1> (*gradient_loss_type)(const Eigen::Tensor<double,1>&,const Eigen::Tensor<double,1>&);
  typedef double (*loss_type)(const Eigen::Tensor<double,1>&,const Eigen::Tensor<double,1>&);

  loss_type GetLoss(Losses loss);
  gradient_loss_type GetGradientOfLoss(Losses loss);

  double Mse(const Eigen::Tensor<double,1>& predicts,const Eigen::Tensor<double,1>& labels);
  double CategoricalCrossEntropy(const Eigen::Tensor<double,1>& logits,const Eigen::Tensor<double,1>& labels);

  Eigen::Tensor<double,1> GradientOfMse(const Eigen::Tensor<double,1>& predicts,const Eigen::Tensor<double,1>& labels);
  Eigen::Tensor<double,1> GradientOfCategoricalCrossEntropy(const Eigen::Tensor<double,1>& logits,const Eigen::Tensor<double,1>& labels);

};//namespace dense
#endif// LOSSES_H
