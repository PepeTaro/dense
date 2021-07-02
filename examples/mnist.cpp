#include "dense.h"
#include "utils/activations.h"
#include "utils/losses.h"

using namespace dense;

#include "../mnist/include/mnist/mnist_reader.hpp"
void InitDataset(Eigen::Tensor<double,2>& training_data,Eigen::Tensor<double,2>& training_labels,int training_data_size,
		 Eigen::Tensor<double,2>& test_data,    Eigen::Tensor<double,2>& test_labels,    int test_data_size     ){

  training_data.setZero();
  training_labels.setZero();
  test_data.setZero();
  test_labels.setZero();
  
  auto dataset = mnist::read_dataset<std::vector,std::vector,uint8_t,uint8_t>("../../mnist/");
  
  for(int data_idx=0;data_idx<training_data_size;++data_idx){
    for(int i=0;i<784;++i){
      training_data(data_idx,i) = (double)dataset.training_images[data_idx][i]/255.0;	
    }    
    training_labels(data_idx,(int)dataset.training_labels[data_idx]) = 1;
  }
  
  for(int data_idx=0;data_idx<test_data_size;++data_idx){
    for(int i=0;i<784;++i){
	test_data(data_idx,i) = (double)dataset.test_images[data_idx][i]/255.0;	
    }
    test_labels(data_idx,(int)dataset.test_labels[data_idx]) = 1;
  }
  
}

void MeasureAccuracy(Dense& network,const Eigen::Tensor<double,2>& data,const Eigen::Tensor<double,2>& labels,int data_size){
  int max_index;
  int correct = 0;
  
  for(int i=0;i<data_size;++i){    
    auto input = data.chip(i,0);
    auto label = labels.chip(i,0);
    auto output = network.Feedforward(input);        
    max_index = Argmax(output);
    if(labels(i,max_index) == 1){      
      correct++;
    }
  }    
  std::cout << "Accuracy(%): " << (100.0)*(double)correct/data_size << std::endl;
  
}


void Train(Dense& network,const Eigen::Tensor<double,2>& training_data,const Eigen::Tensor<double,2>& training_labels,int training_data_size,int epoch){
  double sum_loss;
  Eigen::Tensor<double,1> input;
  Eigen::Tensor<double,1> label;
  Eigen::Tensor<double,1> softmax;
  Eigen::Tensor<double,1> output;

  for(int i=0;i<epoch;++i){
    printf("[Epoch %d]\n:",(i+1));
    sum_loss = 0.0;
    for(int j=0;j<training_data_size;++j){
      
      if(j%100 == 0)
	printf("\rProgress:%f",(1.f*j/training_data_size));
      
      input = training_data.chip(j,0);
      label = training_labels.chip(j,0);
      
      output = network.Feedforward(input);      
      network.Backprop(label);
      network.UpdateGradientOfParameters();
      network.UpdateParameters();
      sum_loss += network.Loss(label);
    }
    printf("\rProgress:%f|",(1.0f));
    printf(" Error:%f\n",sum_loss/training_data_size);
  }
}

void TestMnist(){
  int training_data_size = 60000;
  int test_data_size = 10000;
  
  int input_height = 28;
  int input_width = 28;
  
  //Dense network({784,30,10},{Activations::Relu,Activations::Relu},Losses::CategoricalCrossEntropy);
  //network.SetLearningRate(1e-4);
  Dense network({784,30,10},{Activations::Sigmoid,Activations::Sigmoid},Losses::Mse);
  network.SetLearningRate(0.5);
  
  Eigen::Tensor<double,2> training_data(training_data_size,784);
  Eigen::Tensor<double,2> training_labels(training_data_size,10);
  Eigen::Tensor<double,2> test_data(test_data_size,784);
  Eigen::Tensor<double,2> test_labels(test_data_size,10);

  // データを初期化。
  InitDataset(training_data,training_labels,training_data_size,
	      test_data,test_labels,test_data_size);

  // Epochを30にして学習。
  Train(network,training_data,training_labels,training_data_size,30);

  // 精度測定。
  MeasureAccuracy(network,test_data,test_labels,test_data_size);
}

int main(){
  TestMnist();
  return 0;
}
