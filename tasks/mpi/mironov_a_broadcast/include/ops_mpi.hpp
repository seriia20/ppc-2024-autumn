#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mironov_a_broadcast_mpi {

class ComponentSumPowerBoostImpl : public ppc::core::Task {
 public:
  explicit ComponentSumPowerBoostImpl(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> powers_;
  std::vector<int> result_;
  int delta_ = 0;
  int size_ = 0;
  boost::mpi::communicator world;
};
}  // namespace mironov_a_broadcast_mpi
