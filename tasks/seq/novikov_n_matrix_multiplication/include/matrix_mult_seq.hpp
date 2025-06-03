#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace novikov_n_matrix_multiplication_seq {

std::vector<int> multiply(size_t M, size_t N, size_t K, const std::vector<int>& A, const std::vector<int>& B);

class MatrixMultiplicationSequential : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> A_;
  std::vector<int> B_;
  std::vector<int> C_;
  size_t M_{}, N_{}, K_{};
};

}  // namespace novikov_n_matrix_multiplication_seq
