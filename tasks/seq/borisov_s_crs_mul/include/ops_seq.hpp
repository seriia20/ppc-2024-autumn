// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_crs_mul {

class CrsMatrixMulTask : public ppc::core::Task {
 public:
  explicit CrsMatrixMulTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  const double* A_values_ = nullptr;
  const int* A_col_index_ = nullptr;
  const int* A_row_ptr_ = nullptr;

  const double* B_values_ = nullptr;
  const int* B_col_index_ = nullptr;
  const int* B_row_ptr_ = nullptr;

  double* C_values_ = nullptr;
  int* C_col_index_ = nullptr;
  int* C_row_ptr_ = nullptr;

  int A_nrows_, A_ncols_, A_nnz_;
  int B_nrows_, B_ncols_, B_nnz_;
  int C_nrows_, C_ncols_;
  int C_nnz_;
};

}  // namespace borisov_s_crs_mul