// Copyright 2024 Nesterov Alexander
#include "seq/korotin_e_min_val_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool korotin_e_min_val_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* start = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(start, start + taskData->inputs_count[0], input_.begin());
  // Init value for output
  res = 0.0;
  return true;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  res = input_[0];
  for (unsigned i = 1; i < taskData->inputs_count[0]; i++) {
    if (input_[i] < res) res = input_[i];
  }
  return true;
}

bool korotin_e_min_val_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
