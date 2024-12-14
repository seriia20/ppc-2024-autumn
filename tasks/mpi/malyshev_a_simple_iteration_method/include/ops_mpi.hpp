// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_a_simple_iteration_method/include/matrix.hpp"

namespace malyshev_a_simple_iteration_method_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> C_;
  std::vector<double> B_;
  std::vector<double> D_;
  std::vector<double> X_;
  std::vector<double> X0_;
  double eps_;
  uint32_t n_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> C_;
  std::vector<double> B_;
  std::vector<double> D_;
  std::vector<double> X_;
  std::vector<double> X0_;
  double eps_;
  uint32_t n_;

  boost::mpi::communicator world;
};

}  // namespace malyshev_a_simple_iteration_method_mpi