#include "common_utils.h"

void EXPECT_NEAR_RANK1_TENSOR(const Eigen::Tensor<double,1>& test_tensor,const Eigen::Tensor<double,1>& expected_tensor){
  EXPECT_EQ(test_tensor.dimension(0),expected_tensor.dimension(0)); // 次元が一致しているかチェック
  for(int i=0;i<test_tensor.dimension(0);++i){
    EXPECT_NEAR(test_tensor(i),expected_tensor(i),abs_error);
  }
}
