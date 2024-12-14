// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_m_scalar_product_of_vectors {
std::vector<int> generateRandomVector(int v_size);
int generateRandomNumber(int min, int max);
int calculateDotProduct(const std::vector<int>& vec_1, const std::vector<int>& vec_2);
class VectorDotProduct : public ppc::core::Task {
 public:
  explicit VectorDotProduct(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int res{};
  std::vector<std::vector<int>> input_;
};

}  // namespace koshkin_m_scalar_product_of_vectors