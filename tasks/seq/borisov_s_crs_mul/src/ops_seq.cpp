// Copyright 2024 Nesterov Alexander
#include "seq/borisov_s_crs_mul/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

namespace borisov_s_crs_mul {

bool CrsMatrixMulTask::validation() {
  internal_order_test();

  if (taskData->inputs.size() != 6 || taskData->outputs.size() != 3) {
    return false;
  }

  if (taskData->inputs_count.size() < 6 || taskData->outputs_count.size() < 3) {
    return false;
  }

  A_nnz_ = (int)taskData->inputs_count[0];
  if ((int)taskData->inputs_count[1] != A_nnz_) return false;
  A_nrows_ = (int)taskData->inputs_count[2] - 1;
  if (A_nrows_ < 0) return false;

  B_nnz_ = (int)taskData->inputs_count[3];
  if ((int)taskData->inputs_count[4] != B_nnz_) return false;
  B_nrows_ = (int)taskData->inputs_count[5] - 1;
  if (B_nrows_ < 0) return false;

  {
    const int* A_col = reinterpret_cast<const int*>(taskData->inputs[1]);
    int max_col_A = -1;
    for (int i = 0; i < A_nnz_; i++) {
      if (A_col[i] > max_col_A) max_col_A = A_col[i];
    }
    A_ncols_ = max_col_A + 1;
    if (A_ncols_ <= 0) return false;
  }

  {
    const int* B_col = reinterpret_cast<const int*>(taskData->inputs[4]);
    int max_col_B = -1;
    for (int i = 0; i < B_nnz_; i++) {
      if (B_col[i] > max_col_B) max_col_B = B_col[i];
    }
    B_ncols_ = max_col_B + 1;
    if (B_ncols_ <= 0) return false;
  }

  if (A_ncols_ != B_nrows_) return false;

  if (taskData->outputs_count[2] < (size_t)(A_nrows_ + 1)) {
    return false;
  }

  return true;
}

bool CrsMatrixMulTask::pre_processing() {
  internal_order_test();
  A_values_ = reinterpret_cast<const double*>(taskData->inputs[0]);
  A_col_index_ = reinterpret_cast<const int*>(taskData->inputs[1]);
  A_row_ptr_ = reinterpret_cast<const int*>(taskData->inputs[2]);

  B_values_ = reinterpret_cast<const double*>(taskData->inputs[3]);
  B_col_index_ = reinterpret_cast<const int*>(taskData->inputs[4]);
  B_row_ptr_ = reinterpret_cast<const int*>(taskData->inputs[5]);

  C_values_ = reinterpret_cast<double*>(taskData->outputs[0]);
  C_col_index_ = reinterpret_cast<int*>(taskData->outputs[1]);
  C_row_ptr_ = reinterpret_cast<int*>(taskData->outputs[2]);

  C_nrows_ = A_nrows_;
  C_ncols_ = B_ncols_;

  for (int i = 0; i <= C_nrows_; i++) {
    C_row_ptr_[i] = 0;
  }

  return true;
}

bool CrsMatrixMulTask::run() {
  internal_order_test();

  std::vector<double> temp(C_ncols_, 0.0);
  int nnz_count = 0;

  for (int i = 0; i < C_nrows_; i++) {
    for (int t = 0; t < C_ncols_; t++) {
      temp[t] = 0.0;
    }

    int startA = A_row_ptr_[i];
    int endA = A_row_ptr_[i + 1];

    for (int posA = startA; posA < endA; posA++) {
      double a_val = A_values_[posA];
      int a_col = A_col_index_[posA];

      int startB = B_row_ptr_[a_col];
      int endB = B_row_ptr_[a_col + 1];

      for (int posB = startB; posB < endB; posB++) {
        int b_col = B_col_index_[posB];
        double b_val = B_values_[posB];
        temp[b_col] += a_val * b_val;
      }
    }

    C_row_ptr_[i] = nnz_count;
    for (int col = 0; col < C_ncols_; col++) {
      double val = temp[col];
      if (val != 0.0) {
        C_values_[nnz_count] = val;
        C_col_index_[nnz_count] = col;
        nnz_count++;
      }
    }
  }

  C_row_ptr_[C_nrows_] = nnz_count;
  C_nnz_ = nnz_count;

  return true;
}

bool CrsMatrixMulTask::post_processing() {
  internal_order_test();

  taskData->outputs_count[0] = C_nnz_;
  taskData->outputs_count[1] = C_nnz_;
  taskData->outputs_count[2] = C_nrows_ + 1;

  return true;
}

}  // namespace borisov_s_crs_mul
