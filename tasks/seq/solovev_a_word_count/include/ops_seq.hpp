#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_word_count_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  int res{};
};

}  // namespace solovev_a_word_count_seq
