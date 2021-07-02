#include "dense.h"
#include "utils/activations.h"
#include "utils/losses.h"

#include <iostream>

namespace dense{
  Dense::Dense(std::initializer_list<int> num_of_neurons_in_each_layer,std::initializer_list<Activations> activations,Losses loss):learning_rate_(0.1),weight_decay_(0.0){ 
    InitNumOfLayers(num_of_neurons_in_each_layer.size());
    InitNumOfNeuronsInEachLayer(num_of_neurons_in_each_layer);
    InitActivations(activations);
    InitLoss(loss);
  
    InitWeights(); 
    InitBiases();
    InitGradientOfWeights();    
    InitGradientOfBiases();
  
    InitPreActivations();
    InitPostActivations();  
    InitErrors();
  }

  Dense::~Dense(){

  }
  
  Eigen::Tensor<double,1> Dense::Feedforward(const Eigen::Tensor<double,1>& input){  
    ClearFeedforwardParameters();// pre_activations_,post_activations_を初期化。
    StoreInput(input);

    for(int l=0;l<num_of_layers_;++l){
      pre_activations_[l] = Dot(weights_[l],post_activations_[l]) + biases_[l];    
      post_activations_[l+1] = Activation(pre_activations_[l],l);                
    }
  
    return Output(post_activations_[num_of_layers_]);
  }

  Eigen::Tensor<double,1> Dense::Output(const Eigen::Tensor<double,1>& input){
    // CateogricalCrossEntropyを使用した場合,Outputを確率分布に変換。
    if(loss_ == Losses::CategoricalCrossEntropy){
      return Softmax(input);
    }else{
      return input;
    }
  }

  void Dense::Backprop(const Eigen::Tensor<double,1>& label){
    Eigen::Tensor<double,1> loss;
    int index_of_end_of_layers;
  
    index_of_end_of_layers = num_of_layers_ - 1;
    ClearBackpropParameters();// gradient_of_weights_,gradient_of_baises_,errors_を初期化。
  
    loss = GradientOfLoss(post_activations_[index_of_end_of_layers+1],label);
    errors_[index_of_end_of_layers] = loss*DerivativeOfActivation(pre_activations_[index_of_end_of_layers],index_of_end_of_layers);    

    for(int l = index_of_end_of_layers-1;l>=0;--l){
      errors_[l] = TransposedDot(weights_[l+1],errors_[l+1])*DerivativeOfActivation(pre_activations_[l],l);
    }
  }


  void Dense::UpdateGradientOfParameters(){
    int index_of_end_of_layers;      
    index_of_end_of_layers = num_of_layers_ - 1;

#pragma omp parallel for 
    for(int l = index_of_end_of_layers;l>=0;--l){
      gradient_of_weights_[l] = OuterProduct(errors_[l],post_activations_[l]);
      gradient_of_biases_[l] = errors_[l];        
    }
  }

  void Dense::UpdateParameters(){
    int index_of_end_of_layers;      
    index_of_end_of_layers = num_of_layers_ - 1;
  
#pragma omp parallel for 
    for(int l = index_of_end_of_layers;l>=0;--l){    
      weights_[l] = (1.0 - learning_rate_*weight_decay_)*weights_[l] - learning_rate_*gradient_of_weights_[l];
      biases_[l] = biases_[l] - learning_rate_*gradient_of_biases_[l];
    }
  
  }

  Eigen::Tensor<double,1> Dense::DerivativeOfActivation(const Eigen::Tensor<double,1>& input,int layer_index){
    return input.unaryExpr(GetDerivativeActivation(activations_[layer_index]));    
  }

  Eigen::Tensor<double,1> Dense::Activation(const Eigen::Tensor<double,1>& input,int layer_index){
    return input.unaryExpr(GetActivation(activations_[layer_index]));    
  }

  Eigen::Tensor<double,1> Dense::GradientOfLoss(const Eigen::Tensor<double,1>& predicts,const Eigen::Tensor<double,1>& labels){    
    return (GetGradientOfLoss(loss_))(predicts,labels);
  }

  double Dense::Loss(const Eigen::Tensor<double,1>& labels){
    return (GetLoss(loss_))(post_activations_[num_of_layers_],labels);
  }

  void Dense::InitNumOfLayers(int num_of_layers){
    if(num_of_layers < 2){ // 少なくとも、2つのレイヤー(InputとOutputレイヤー)は必要。
      std::cerr << "[!]Error(InitNumOfLayers): レイヤーを少なくとも2つは指定してください。" << std::endl;
      exit(-1);
    }
  
    num_of_layers_ = num_of_layers - 1;// Input Layerは含めない。
  }

  void Dense::InitNumOfNeuronsInEachLayer(const std::initializer_list<int>& num_of_neurons_in_each_layer){
    for(auto num_of_neurons : num_of_neurons_in_each_layer) 
      num_of_neurons_in_each_layer_.push_back(num_of_neurons);  
  }

  void Dense::InitActivations(std::initializer_list<Activations> activations){
    if(activations.size() != num_of_layers_){ 
      std::cerr << "[!]Error(InitActivations): Activation関数の数とInputを除いたレイヤー数が等しくありません。" << std::endl;
      exit(-1);
    }
  
    for(auto activation : activations) 
      activations_.push_back(activation);
  }

  void Dense::InitLoss(Losses loss){
    loss_ = loss;
  }

  void Dense::InitWeights(){
    Eigen::Tensor<double,2> weight;
    int input_size;
    int output_size;
    
    for(int l=0;l<num_of_layers_;++l){
      input_size = num_of_neurons_in_each_layer_[l];
      output_size = num_of_neurons_in_each_layer_[l+1];


      weight = Eigen::Tensor<double,2>(output_size,input_size);
    
      if(activations_[l] == Activations::Relu){
	weight.setRandom(); //setRandomは,正乱数を返す。
      }else if(activations_[l] == Activations::Sigmoid){
	weight.setRandom<Eigen::internal::NormalRandomGenerator<double>>();
      }
      weights_.push_back(weight);
    }  
  }
    
  void Dense::InitBiases(){      
    Eigen::Tensor<double,1> bias;
    int output_size;
    
    for(int l=0;l<num_of_layers_;++l){
      output_size = num_of_neurons_in_each_layer_[l+1];
    
      bias = Eigen::Tensor<double,1>(output_size);
      bias.setZero();
      biases_.push_back(bias);
    }  
  }

  void Dense::InitGradientOfWeights(){
    Eigen::Tensor<double,2> gradient_of_weight;
    int input_size;
    int output_size;
    
    for(int l=0;l<num_of_layers_;++l){
      input_size = num_of_neurons_in_each_layer_[l];
      output_size = num_of_neurons_in_each_layer_[l+1];
    
      gradient_of_weight = Eigen::Tensor<double,2>(output_size,input_size);
      gradient_of_weight.setZero();
      gradient_of_weights_.push_back(gradient_of_weight);
    }
  }

  void Dense::InitGradientOfBiases(){
    Eigen::Tensor<double,1> gradient_of_bias;
    int output_size;
    
    for(int l=0;l<num_of_layers_;++l){
      output_size = num_of_neurons_in_each_layer_[l+1];
    
      gradient_of_bias = Eigen::Tensor<double,1>(output_size);
      gradient_of_bias.setZero();
      gradient_of_biases_.push_back(gradient_of_bias);
    }
  }

  void Dense::InitPreActivations(){
    Eigen::Tensor<double,1> pre_activation;
    int output_size;
    
    for(int l=0;l<num_of_layers_;++l){
      output_size = num_of_neurons_in_each_layer_[l+1];
    
      pre_activation = Eigen::Tensor<double,1>(output_size);
      pre_activation.setZero();
      pre_activations_.push_back(pre_activation);
    }
  }

  void Dense::InitPostActivations(){
    Eigen::Tensor<double,1> post_activation;
    int output_size;

    // Inputのサイズも含めるため,pre_activations_のサイズより1大きい。
    for(int l=0;l<num_of_layers_+1;++l){
      output_size = num_of_neurons_in_each_layer_[l];
    
      post_activation = Eigen::Tensor<double,1>(output_size);
      post_activation.setZero();
      post_activations_.push_back(post_activation);
    }
  }

  void Dense::InitErrors(){
    Eigen::Tensor<double,1> error;
    int output_size;
    
    for(int l=0;l<num_of_layers_;++l){
      output_size = num_of_neurons_in_each_layer_[l+1];
    
      error = Eigen::Tensor<double,1>(output_size);
      error.setZero();
      errors_.push_back(error);
    }  
  }

  void Dense::ClearFeedforwardParameters(){
    for(int l=0;l<num_of_layers_;++l){    
      pre_activations_[l].setZero();
      post_activations_[l].setZero();
    }
    // post_activations_は,inputデータを先頭に含むため,pre_activations_より1つサイズが大きい。
    post_activations_[num_of_layers_].setZero();
  }

  void Dense::ClearBackpropParameters(){
    for(int l=0;l<num_of_layers_;++l){
      gradient_of_weights_[l].setZero();
      gradient_of_biases_[l].setZero();
      errors_[l].setZero();
    }  
  }

  int Dense::GetNumOfNeuronsInLayer(int layer_index) const{
    if(layer_index < 0 or layer_index > num_of_layers_){ 
      std::cerr << "[!]Error(GetNumOfNeuronsInLayer): layer_indexが正しくありません。" << std::endl;
      exit(-1);
    }

    return num_of_neurons_in_each_layer_[layer_index];
  }

  int Dense::GetNumOfLayers() const{
    return num_of_layers_;
  }

  double Dense::GetLearningRate() const{
    return learning_rate_;
  }

  double Dense::GetWeightDecay() const{
    return weight_decay_;
  }

  void Dense::StoreInput(const Eigen::Tensor<double,1>& input){
    post_activations_[0] = input;
  }

  void Dense::SetLearningRate(double learning_rate){
    learning_rate_ = learning_rate;
  }

  void Dense::SetWeightDecay(double weight_decay){
    weight_decay_ = weight_decay;
  }

};//namespace dense
