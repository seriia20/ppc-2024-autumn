// Copyright 2024 Khovansky Dmitry
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace khovansky_d_max_of_vector_elements_seq {

class MaxOfVectorSeq : public ppc::core::Task {
 public:
  explicit MaxOfVectorSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
};
}  // namespace khovansky_d_max_of_vector_elements_seq