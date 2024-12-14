// Copyright 2024 Nesterov Alexander
#pragma once
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sidorina_p_check_lexicographic_order_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<char>> input_;
  int res{};
};
}  // namespace sidorina_p_check_lexicographic_order_seq