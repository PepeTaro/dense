#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <vector>
#include <cassert>
#include <initializer_list>
#include <ctime>
#include <cstdlib>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "utils/utils.h"
#include "utils/activations.h"
#include "utils/losses.h"

namespace dense{
  class Dense{
  public: 
    int num_of_layers_; // ネットワークのレイヤー数、Inputレイヤーを含めないことに注意。
    double learning_rate_;
    double weight_decay_;
  
    std::vector<int> num_of_neurons_in_each_layer_; // 各レイヤーにおけるニューロンの数。
    std::vector<Eigen::Tensor<double,2>> weights_;
    std::vector<Eigen::Tensor<double,1>> biases_;
    std::vector<Eigen::Tensor<double,2>> gradient_of_weights_;
    std::vector<Eigen::Tensor<double,1>> gradient_of_biases_;

    std::vector<Activations> activations_;
    Losses loss_;
  
    std::vector<Eigen::Tensor<double,1>> pre_activations_; // Backpropagation時に使用する
    std::vector<Eigen::Tensor<double,1>> post_activations_; // Backpropagation時に使用する。
    std::vector<Eigen::Tensor<double,1>> errors_; // Backpropagation時に使用する。
        
  public:    
    Dense(std::initializer_list<int> num_of_neurons_in_each_layer,std::initializer_list<Activations> activations,Losses loss);
    virtual ~Dense();

    Eigen::Tensor<double,1> Feedforward(const Eigen::Tensor<double,1>& input);
    Eigen::Tensor<double,1> Output(const Eigen::Tensor<double,1>& input);
    Eigen::Tensor<double,1> DerivativeOfActivation(const Eigen::Tensor<double,1>& input,int layer_index);
    Eigen::Tensor<double,1> Activation(const Eigen::Tensor<double,1>& input,int layer_index);
    Eigen::Tensor<double,1> GradientOfLoss(const Eigen::Tensor<double,1>& predicts,const Eigen::Tensor<double,1>& labels);
    double Loss(const Eigen::Tensor<double,1>& labels);
  
    void Backprop(const Eigen::Tensor<double,1>& label);
    void UpdateGradientOfParameters();
    void UpdateParameters();
  
    void InitNumOfLayers(int num_of_layers);
    void InitNumOfNeuronsInEachLayer(const std::initializer_list<int>& num_of_neurons_in_each_layer);
    void InitActivations(std::initializer_list<Activations> activations);
    void InitLoss(Losses loss);
    void InitWeights();    
    void InitBiases();
    void InitGradientOfWeights();    
    void InitGradientOfBiases();

    void InitPreActivations();
    void InitPostActivations();
    void InitErrors();

    void ClearFeedforwardParameters();
    void ClearBackpropParameters();

    int GetNumOfNeuronsInLayer(int layer_index) const;
    int GetNumOfLayers() const;
    double GetLearningRate() const;
    double GetWeightDecay() const;

    void StoreInput(const Eigen::Tensor<double,1>& input);
    void SetLearningRate(double learning_rate);
    void SetWeightDecay(double weight_decay);
    
  };//class Dnese
  
};//namespace dense
#endif// DENSE_H
