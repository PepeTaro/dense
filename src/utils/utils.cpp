#include "utils/utils.h"

namespace dense{
  int Argmax(const Eigen::Tensor<double,1>& vector){
    int max_index;
    double max;
  
    max_index = 0;
    max = vector(0);
    for(int i=1;i<vector.dimension(0);++i){
      if(vector(i) > max){
	max = vector(i);
	max_index = i;
      }
    }
    return max_index;
  }

  double Max(const Eigen::Tensor<double,1>& vector){
    double max;

    max = ((Eigen::Tensor<double,0>)vector.maximum())(0);
    return max;  
  }

  double Sum(const Eigen::Tensor<double,1>& vector){
    double sum;

    sum = ((Eigen::Tensor<double,0>)vector.sum())(0);
    return sum;
  }

  // matrix*vectorを計算して,その結果を返す。(ここで,*は行列積を意味する)
  Eigen::Tensor<double,1> Dot(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,1>& vector){
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims;
  
    product_dims = { Eigen::IndexPair<int>(1, 0)};// matrixのaxis1(列)とvectorのaxis0(行)を掛け合わせることを意味する。
    return matrix.contract(vector,product_dims);
  }

  // ２つ目の引数は,design matrixなので,行がデータのインデックスを表し,列がデータのfeaturesを表すことに注意,
  // つまり,matrxiとdesign matrixを行列積する前に,design matrixを転置することに注意。
  Eigen::Tensor<double,2> Dot(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,2>& design_matrix){    
    Eigen::array<Eigen::IndexPair<int>,1> product_dims;
    product_dims = {Eigen::IndexPair<int>(1,1)};// design matrixを転置して行列積をすることを意味する。  

    return matrix.contract(design_matrix,product_dims);
  }

  // matrixを転置してから,その転置行列とvectorの行列積を計算してその結果を返す。
  Eigen::Tensor<double,1> TransposedDot(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,1>& vector){
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims;
  
    product_dims = {Eigen::IndexPair<int>(0,0)};// matrixのaxis0(行)とvectorのaxis0(行)を掛け合わせることを意味する。
    return matrix.contract(vector,product_dims);
  }

  // 2つのベクトルの外積を計算し,その結果を返す。
  Eigen::Tensor<double,2> OuterProduct(const Eigen::Tensor<double,1>& vec1,const Eigen::Tensor<double,1>& vec2){
    Eigen::array<Eigen::IndexPair<int>,0> product_dims;
  
    product_dims = {};// 空にすると外積ができる。(Eigenの仕様)
    return vec1.contract(vec2, product_dims);
  }

  Eigen::Tensor<double,2> Broadcast(const Eigen::Tensor<double,2>& matrix,const Eigen::Tensor<double,1>& vector){
    std::array<int,1> copies;
    std::array<int,2> shape;
    int num_of_copies;
    int matrix_row,matrix_col;

    matrix_row = matrix.dimension(0);
    matrix_col = matrix.dimension(1);
    num_of_copies = matrix_col;

    copies = std::array<int,1>{num_of_copies};
    shape = std::array<int,2>{matrix_row,matrix_col};
  
    return (matrix + vector.broadcast(copies).reshape(shape));
  }

  Eigen::Tensor<double,1> Flatten(const Eigen::Tensor<double,3>& tensor,int dimension){
    std::array<int,1> shape;
  
    shape = std::array<int,1> {dimension};  
    return tensor.reshape(shape);
  }

};//namespace dense
