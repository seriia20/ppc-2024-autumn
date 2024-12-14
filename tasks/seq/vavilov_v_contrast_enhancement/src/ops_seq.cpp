#include "seq/vavilov_v_contrast_enhancement/include/ops_seq.hpp"

bool vavilov_v_contrast_enhancement_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  size_t data_size = taskData->inputs_count[0];
  input_.resize(data_size);
  int* data = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < data_size; i++) {
    input_[i] = data[i];
  }

  output_.resize(data_size, 0);
  return true;
}

bool vavilov_v_contrast_enhancement_seq::TestTaskSequential::validation() {
  internal_order_test();

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->outputs_count[0] == taskData->inputs_count[0]));
}

bool vavilov_v_contrast_enhancement_seq::TestTaskSequential::run() {
  internal_order_test();

  p_min_ = *std::min_element(input_.begin(), input_.end());
  p_max_ = *std::max_element(input_.begin(), input_.end());

  if (p_max_ == p_min_) {
    std::fill(output_.begin(), output_.end(), 0);
    return true;
  }

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<int>(static_cast<double>(input_[i] - p_min_) * 255 / (p_max_ - p_min_));
  }
  return true;
}

bool vavilov_v_contrast_enhancement_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
