// Copyright 2024 Nesterov Alexander
#include "seq/borisov_s_crs_mul/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

namespace borisov_s_crs_mul {

bool CrsMatrixMulTask::validation() {
  internal_order_test();

  std::cout << "TEST0" << std::endl;

  if (taskData->inputs.size() != 6 || taskData->outputs.size() != 3) {
    return false;
  }

  if (taskData->inputs_count.size() < 6 || taskData->outputs_count.size() < 3) {
    return false;
  }

  int A_nrows = static_cast<int>(taskData->inputs_count[2] - 1);
  int B_nrows = static_cast<int>(taskData->inputs_count[5] - 1);

  std::cout << "TEST-1" << std::endl;

  return A_nrows > 0 && B_nrows > 0;
}

bool CrsMatrixMulTask::pre_processing() {
  internal_order_test();

  std::cout << "TEST2" << std::endl;

  A_nnz_ = static_cast<int>(taskData->inputs_count[0]);
  A_values_.assign(reinterpret_cast<const double*>(taskData->inputs[0]),
                   reinterpret_cast<const double*>(taskData->inputs[0]) + A_nnz_);
  A_col_index_.assign(reinterpret_cast<const int*>(taskData->inputs[1]),
                      reinterpret_cast<const int*>(taskData->inputs[1]) + A_nnz_);
  A_row_ptr_.assign(reinterpret_cast<const int*>(taskData->inputs[2]),
                    reinterpret_cast<const int*>(taskData->inputs[2]) + taskData->inputs_count[2]);
  A_nrows_ = static_cast<int>(taskData->inputs_count[2] - 1);
  A_ncols_ = *std::max_element(A_col_index_.begin(), A_col_index_.end()) + 1;

  B_nnz_ = static_cast<int>(taskData->inputs_count[3]);
  B_values_.assign(reinterpret_cast<const double*>(taskData->inputs[3]),
                   reinterpret_cast<const double*>(taskData->inputs[3]) + B_nnz_);
  B_col_index_.assign(reinterpret_cast<const int*>(taskData->inputs[4]),
                      reinterpret_cast<const int*>(taskData->inputs[4]) + B_nnz_);
  B_row_ptr_.assign(reinterpret_cast<const int*>(taskData->inputs[5]),
                    reinterpret_cast<const int*>(taskData->inputs[5]) + taskData->inputs_count[5]);
  B_nrows_ = static_cast<int>(taskData->inputs_count[5] - 1);
  B_ncols_ = *std::max_element(B_col_index_.begin(), B_col_index_.end()) + 1;

  C_nrows_ = A_nrows_;
  C_ncols_ = B_ncols_;
  C_row_ptr_.assign(C_nrows_ + 1, 0);
  C_values_.clear();
  C_col_index_.clear();

  return true;
}

bool CrsMatrixMulTask::run() {
  internal_order_test();

  std::cout << "TEST" << std::endl;

  std::vector<double> temp(C_ncols_, 0.0);

  for (int i = 0; i < C_nrows_; ++i) {
    std::fill(temp.begin(), temp.end(), 0.0);

    int startA = A_row_ptr_[i];
    int endA = A_row_ptr_[i + 1];

    for (int posA = startA; posA < endA; ++posA) {
      double a_val = A_values_[posA];
      int a_col = A_col_index_[posA];

      int startB = B_row_ptr_[a_col];
      int endB = B_row_ptr_[a_col + 1];

      for (int posB = startB; posB < endB; ++posB) {
        int b_col = B_col_index_[posB];
        double b_val = B_values_[posB];
        temp[b_col] += a_val * b_val;
      }
    }

    C_row_ptr_[i] = static_cast<int>(C_values_.size());
    for (int col = 0; col < C_ncols_; ++col) {
      if (temp[col] != 0.0) {
        C_values_.push_back(temp[col]);
        C_col_index_.push_back(col);
      }
    }
  }

  C_row_ptr_[C_nrows_] = static_cast<int>(C_values_.size());
  C_nnz_ = static_cast<int>(C_values_.size());

  return true;
}

bool CrsMatrixMulTask::post_processing() {
  internal_order_test();

  taskData->outputs_count[0] = static_cast<unsigned int>(C_values_.size());
  taskData->outputs_count[1] = static_cast<unsigned int>(C_col_index_.size());
  taskData->outputs_count[2] = static_cast<unsigned int>(C_row_ptr_.size());

  taskData->outputs[0] = reinterpret_cast<uint8_t*>(C_values_.data());
  taskData->outputs[1] = reinterpret_cast<uint8_t*>(C_col_index_.data());
  taskData->outputs[2] = reinterpret_cast<uint8_t*>(C_row_ptr_.data());

  std::cout << "Post-processing complete:\n";
  std::cout << "Matrix C:\n";
  std::cout << "  Values: ";
  for (const auto& val : C_values_) {
    std::cout << val << " ";
  }
  std::cout << "\n";

  std::cout << "  Column indices: ";
  for (const auto& col : C_col_index_) {
    std::cout << col << " ";
  }
  std::cout << "\n";

  std::cout << "  Row pointers: ";
  for (const auto& row : C_row_ptr_) {
    std::cout << row << " ";
  }
  std::cout << "\n";

  return true;
}

}  // namespace borisov_s_crs_mul
