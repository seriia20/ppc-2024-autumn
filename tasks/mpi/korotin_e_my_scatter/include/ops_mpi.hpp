#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_my_scatter_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_;
  double res = 0.0;
};

class TestMPITaskMyParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskMyParallel(std::shared_ptr<ppc::core::TaskData> taskData_, int root_)
      : Task(std::move(taskData_)), root(root_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static int MPI_My_Scatter(void* send_buf, int sendcount, MPI_Datatype sendtype, void* recv_buf, int recvcount,
                            MPI_Datatype recvtype, int src, MPI_Comm comm);

 private:
  std::vector<double> input_, local_input_;
  double res = 0.0;
  int root;
  boost::mpi::communicator world;
};

}  // namespace korotin_e_my_scatter_mpi
