#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_lattice_torus {

class GridTorusTopologyParallel : public ppc::core::Task {
 public:
  explicit GridTorusTopologyParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
};

}  // namespace mezhuev_m_lattice_torus