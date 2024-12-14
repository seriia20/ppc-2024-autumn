#pragma once
#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iterator>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
namespace kudryashova_i_gather_my {
int8_t vectorDotProductGather(const std::vector<int8_t>& vector1, const std::vector<int8_t>& vector2);
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_data;
  std::vector<int> firstHalf, secondHalf;
  int reference{};
};
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <typename T>
  void gather_my(const boost::mpi::communicator& wrld, const T& local_data, std::vector<T>& full_result, int root);

 private:
  boost::mpi::communicator world;
  std::vector<int> input_data;
  std::vector<int> local_input1_, local_input2_;
  std::vector<int> firstHalf, secondHalf;
  std::vector<int> segments;
  int result{};
  int local_result;
  int delta{};
  int processes{};
};
}  // namespace kudryashova_i_gather_my