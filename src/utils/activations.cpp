#include "utils/activations.h"

namespace dense{
  activation_type GetActivation(Activations activation){
    switch(activation){
    case Activations::Relu : return Relu;
    case Activations::Sigmoid : return Sigmoid;
    default:
      std::cerr << "[!]Error(GetActivation): activationが正しく指定されていません。" << std::endl;
      exit(-1);
    }
  }

  activation_type GetDerivativeActivation(Activations activation){
    switch(activation){
    case Activations::Relu : return DerivativeOfRelu;
    case Activations::Sigmoid : return DerivativeOfSigmoid;
    default:
      std::cerr << "[!]Error(GetDerivativeActivation): activationが正しく指定されていません。" << std::endl;
      exit(-1);    
    }  
  }

  double Relu(double x){
    if(x >= 0)
      return x;
    else
      return 0;
  }

  double Sigmoid(double z){
    return 1.0/(1.0+exp(-z));
  }

  double DerivativeOfSigmoid(double z){
    return Sigmoid(z)*(1.0 - Sigmoid(z));  
  }

  double DerivativeOfRelu(double x){
    if(x >= 0)
      return 1;
    else
      return 0;
  }

  // Murphy 2021 p52参照
  double LogSumExp(const Eigen::Tensor<double,1>& logits){
    Eigen::Tensor<double,1> logits_minus_max;
    Eigen::Tensor<double,1> exp_logits_minus_max;
    Eigen::Tensor<double,1> max_vector;
    int dim_of_logits;
    double sum_of_exps;
    double return_value;
    double max;

    dim_of_logits = logits.dimension(0);
  
    max_vector = Eigen::Tensor<double,1>(dim_of_logits);
    max = Max(logits);
    max_vector.setConstant(max);

    logits_minus_max = logits - max_vector;
    exp_logits_minus_max = logits_minus_max.exp();
  
    sum_of_exps  = Sum(exp_logits_minus_max);
    return_value = max + log(sum_of_exps);

    return return_value;
  }

  // Murphy 2021 p52参照
  Eigen::Tensor<double,1> Softmax(const Eigen::Tensor<double,1>& logits){
    Eigen::Tensor<double,1> softmax;
    Eigen::Tensor<double,1> lse_vector;
    int dim_of_logits;
    double lse;
  
    dim_of_logits = logits.dimension(0);

    lse = LogSumExp(logits);
    lse_vector = Eigen::Tensor<double,1>(dim_of_logits);  
    lse_vector.setConstant(lse);

    softmax = (logits - lse_vector).exp();
  
    return softmax;
  }

};//namespace dense
