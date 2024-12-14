// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_readers_and_writers/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool zinoviev_a_readers_and_writers_seq::ReadersWritersTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool zinoviev_a_readers_and_writers_seq::ReadersWritersTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool zinoviev_a_readers_and_writers_seq::ReadersWritersTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_; i++) {
    res++;
  }
  return true;
}

bool zinoviev_a_readers_and_writers_seq::ReadersWritersTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
