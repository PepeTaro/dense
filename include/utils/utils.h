#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace dense{
  int Argmax(const Eigen::Tensor<double,1>& vector);
  double Max(const Eigen::Tensor<double,1>& vector);
  double Sum(const Eigen::Tensor<double,1>& vector);

  Eigen::Tensor<double,1> Dot(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,1>& vector);
  Eigen::Tensor<double,2> Dot(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,2>& design_matrix);
  Eigen::Tensor<double,1> TransposedDot(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,1>& vector);
  Eigen::Tensor<double,2> OuterProduct(const Eigen::Tensor<double,1>& vec1,const Eigen::Tensor<double,1>& vec2);
  Eigen::Tensor<double,2> Broadcast(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,1>& vector);
  Eigen::Tensor<double,1> Flatten(const Eigen::Tensor<double,3>& tensor,int dimension);

};//namespace dense
#endif// UTILS_H
