#include "seq/borisov_s_sum_of_rows/include/ops_seq.hpp"

using namespace std::chrono_literals;

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential::pre_processing() {
  internal_order_test();

  size_t rows = taskData->inputs_count[0];
  size_t cols = taskData->inputs_count[1];

  if (rows <= 0 || cols <= 0) {
    return false;
  }

  int* data = reinterpret_cast<int*>(taskData->inputs[0]);
  if (data == nullptr) {
    return false;
  }

  matrix_.resize(rows * cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      matrix_[(i * cols) + j] = data[(i * cols) + j];
    }
  }

  row_sums_.resize(rows, 0);
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential::validation() {
  internal_order_test();

  if (taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }

  size_t cols = taskData->inputs_count.size() > 1 ? taskData->inputs_count[1] : 0;
  if (cols <= 0) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential::run() {
  internal_order_test();

  size_t rows = taskData->inputs_count[0];
  size_t cols = taskData->inputs_count[1];

  if (!matrix_.empty() && row_sums_.size() == rows) {
    for (size_t i = 0; i < rows; i++) {
      int row_sum = 0;
      for (size_t j = 0; j < cols; j++) {
        row_sum += matrix_[(i * cols) + j];
      }
      row_sums_[i] = row_sum;
    }
  }
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::post_processing() {
  internal_order_test();

  if (!row_sums_.empty()) {
    int* out = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t i = 0; i < row_sums_.size(); i++) {
      out[i] = row_sums_[i];
    }
  }
  return true;
}
