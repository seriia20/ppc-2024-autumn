#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_allreduce_mine_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputData_;
  std::vector<int> resultData_;
  int columnCount{};
  int rowCount{};
  std::vector<int> countAboveMin_;
};

class TestMPITaskMyOwnParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskMyOwnParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <typename T>
  void my_all_reduce(const boost::mpi::communicator& world, const T* in_values, T* out_values, int n);

 private:
  std::vector<int> inputData_;
  std::vector<int> resultData_;
  std::vector<int> countAboveMin_;
  int columnCount{};
  int rowCount{};
  boost::mpi::communicator world;
};

}  // namespace plekhanov_d_allreduce_mine_mpi