// Copyright 2023 Nesterov Alexander
#include "mpi/petrov_a_nearest_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  min_distance_ = std::numeric_limits<int>::max();
  closest_pair_ = {0, 1};
  return true;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size() - 1; i++) {
    int distance = abs(input_[i + 1] - input_[i]);
    if (distance < min_distance_) {
      min_distance_ = distance;
      closest_pair_ = {input_[i], input_[i + 1]};
    }
  }
  return true;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = closest_pair_.first;
  reinterpret_cast<int*>(taskData->outputs[0])[1] = closest_pair_.second;
  return true;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }

  min_distance_ = std::numeric_limits<int>::max();
  return true;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] < 2) {
      return false;
    }
    return taskData->outputs_count[0] == 2;
  }
  return true;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::pair<int, int> local_pair;

  unsigned int delta = 0;
  unsigned int dop = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    dop = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);

  int local_min_distance = std::numeric_limits<int>::max();

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size() - 1; proc++) {
      world.send(proc, 0, input_.data() + proc * delta + dop, delta + 1);
    }
    if (world.size() != 1) {
      world.send(world.size() - 1, 0, input_.data() + dop + (world.size() - 1) * delta, delta);
    }
  }
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(delta + dop + ((world.size() == 1) ? 0 : 1));
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + dop + delta + ((world.size() == 1) ? 0 : 1));
  } else if (world.rank() < world.size() - 1) {
    local_input_ = std::vector<int>(delta + 1);
    world.recv(0, 0, local_input_.data(), delta + 1);
  } else {
    local_input_ = std::vector<int>(delta);
    world.recv(0, 0, local_input_.data(), delta);
  }

  for (size_t i = 0; i < local_input_.size() - 1; i++) {
    int distance = abs(local_input_[i + 1] - local_input_[i]);
    if (distance < local_min_distance) {
      local_min_distance = distance;
      local_pair = {local_input_[i], local_input_[i + 1]};
    }
  }

  std::pair<int, int> global_pair;
  reduce(
      world, local_pair, global_pair,
      [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return (std::abs(a.second - a.first) < std::abs(b.second - b.first)) ? a : b;
      },
      0);

  if (world.rank() == 0) {
    closest_pair_ = global_pair;
  }

  return true;
}

bool petrov_a_nearest_neighbor_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = closest_pair_.first;
    reinterpret_cast<int*>(taskData->outputs[0])[1] = closest_pair_.second;
  }
  return true;
}