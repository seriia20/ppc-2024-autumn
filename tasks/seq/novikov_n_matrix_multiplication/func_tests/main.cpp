#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/novikov_n_matrix_multiplication/include/matrix_mult_seq.hpp"

TEST(novikov_n_matrix_multiplication_seq, simple_2x2) {
  std::vector<int> dimsA = {2, 2};
  std::vector<int> dimsB = {2, 2};
  std::vector<int> A = {1, 2, 3, 4};
  std::vector<int> B = {1, 0, 0, 1};
  std::vector<int> out(4, 0);
  std::vector<int> ans = {1, 2, 3, 4};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsA.data()));
  taskData->inputs_count.emplace_back(dimsA.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsB.data()));
  taskData->inputs_count.emplace_back(dimsB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(B.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(out, ans);
}

TEST(novikov_n_matrix_multiplication_seq, invalid_sizes) {
  std::vector<int> dimsA = {2, 3};
  std::vector<int> dimsB = {4, 2};
  std::vector<int> A(6, 1);
  std::vector<int> B(8, 1);
  std::vector<int> out(4, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsA.data()));
  taskData->inputs_count.emplace_back(dimsA.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsB.data()));
  taskData->inputs_count.emplace_back(dimsB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(B.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(novikov_n_matrix_multiplication_seq, rectangular) {
  std::vector<int> dimsA = {2, 3};
  std::vector<int> dimsB = {3, 2};
  std::vector<int> A = {1, 2, 3, 4, 5, 6};
  std::vector<int> B = {7, 8, 9, 10, 11, 12};
  std::vector<int> out(4, 0);
  std::vector<int> ans = {58, 64, 139, 154};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsA.data()));
  taskData->inputs_count.emplace_back(dimsA.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimsB.data()));
  taskData->inputs_count.emplace_back(dimsB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(B.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  novikov_n_matrix_multiplication_seq::MatrixMultiplicationSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(out, ans);
}
