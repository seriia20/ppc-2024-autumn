// Copyright 2024 Sdobnov Vladimir
#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace Sdobnov_V_sum_of_vector_elements {

int vec_elem_sum(const std::vector<int>& vec);

class SumVecElemSequential : public ppc::core::Task {
 public:
  explicit SumVecElemSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res_{};
};

class SumVecElemParallel : public ppc::core::Task {
 public:
  explicit SumVecElemParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res_{};
  boost::mpi::communicator world;
};

}  // namespace Sdobnov_V_sum_of_vector_elements