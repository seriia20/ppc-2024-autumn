#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_contrast_enhancement_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  uint8_t p_min_{0}, p_max_{255};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> local_input_;
  std::vector<uint8_t> output_;
  uint8_t p_min_global_{0}, p_max_global_{255};
  uint8_t p_min_local_{0}, p_max_local_{255};
  boost::mpi::communicator world;
};

}  // namespace vavilov_v_contrast_enhancement_mpi
