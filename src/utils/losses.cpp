#include "utils/losses.h"

namespace dense{
  loss_type GetLoss(Losses loss){
    switch(loss){
    case Losses::Mse : return Mse;
    case Losses::CategoricalCrossEntropy : return CategoricalCrossEntropy;
    default:
      std::cerr << "[!]Error(GetLoss): lossが正しく指定されていません。" << std::endl;
      exit(-1);
    }
  }

  gradient_loss_type GetGradientOfLoss(Losses loss){
    switch(loss){
    case Losses::Mse : return GradientOfMse;
    case Losses::CategoricalCrossEntropy : return GradientOfCategoricalCrossEntropy;
    default:
      std::cerr << "[!]Error(GetGradientOfLoss): lossが正しく指定されていません。" << std::endl;
      exit(-1);
    }
  }

  double Mse(const Eigen::Tensor<double,1>& predicts,const Eigen::Tensor<double,1>& labels){
    Eigen::Tensor<double,1> squared_errors;
    double mse;

    squared_errors = (predicts - labels).pow(2);
    mse = 0.5*Sum(squared_errors);
    return mse;
  }

  //https://gombru.github.io/2018/05/23/cross_entropy_loss/
  double CategoricalCrossEntropy(const Eigen::Tensor<double,1>& logits,const Eigen::Tensor<double,1>& labels){
    Eigen::Tensor<double,1> softmax;
    Eigen::Tensor<double,1> log_of_softmax;
    Eigen::Tensor<double,1> labels_times_log_of_softmax;
    double loss;
  
    softmax = Softmax(logits);
    log_of_softmax = softmax.log();
    labels_times_log_of_softmax = labels*log_of_softmax;// 成分ごとに掛け合わせていることに注意。
    loss = -1.0 * Sum(labels_times_log_of_softmax);
  
    return loss;
  }

  Eigen::Tensor<double,1> GradientOfMse(const Eigen::Tensor<double,1>& predicts,const Eigen::Tensor<double,1>& labels){
    return (predicts - labels);
  }

  //https://gombru.github.io/2018/05/23/cross_entropy_loss/
  Eigen::Tensor<double,1> GradientOfCategoricalCrossEntropy(const Eigen::Tensor<double,1>& logits,const Eigen::Tensor<double,1>& labels){
    Eigen::Tensor<double,1> softmax;
  
    softmax = Softmax(logits);  
    return (softmax - labels);
  }

};//namespace dense
