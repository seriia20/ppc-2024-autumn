#include "mpi/kudryashova_i_gather/include/GatherMPI.hpp"

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
namespace kudryashova_i_gather {

int8_t vectorDotProductGather(const std::vector<int8_t>& vector1, const std::vector<int8_t>& vector2) {
  int result = 0;
  for (unsigned long i = 0; i < vector1.size(); i++) {
    result += vector1[i] * vector2[i];
  }
  return result;
}

bool kudryashova_i_gather::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_data.resize(taskData->inputs_count[0]);
  if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
    return false;
  }
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_data.begin());
  size_t count = taskData->inputs_count[0];
  size_t halfSize = count / 2;
  firstHalf.resize(halfSize);
  secondHalf.resize(count - halfSize);
  std::copy(input_data.begin(), input_data.begin() + halfSize, firstHalf.begin());
  std::copy(input_data.begin() + halfSize, input_data.begin() + count, secondHalf.begin());
  return true;
}

bool kudryashova_i_gather::TestMPITaskSequential::validation() {
  internal_order_test();
  return ((taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 1) &&
          taskData->outputs_count[0] == 1 && taskData->outputs.size() == taskData->outputs_count.size() &&
          taskData->inputs_count[0] > 1 && taskData->inputs_count[0] % 2 == 0);
}

bool kudryashova_i_gather::TestMPITaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < taskData->inputs_count[0] / 2; ++i) {
    reference += firstHalf[i] * secondHalf[i];
  }
  return true;
}

bool kudryashova_i_gather::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = reference;
  return true;
}

bool kudryashova_i_gather::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int remainder = 0;
  processes = world.size();
  int counting_proc = world.size() - 1;
  if (world.rank() == 0) {
    if (processes == 1 || (int)(taskData->inputs_count[0]) < processes) {
      delta = (taskData->inputs_count[0]) / 2;
    } else {
      delta = (taskData->inputs_count[0] / 2) / counting_proc;
      remainder = (taskData->inputs_count[0] / 2) % counting_proc;
    }
  }
  segments.resize(processes);
  if (world.rank() == 0) {
    segments[0] = delta;
    for (int i = 1; i < processes; ++i) {
      segments[i] = delta + (i <= remainder ? 1 : 0);
    }
  }
  if (world.rank() == 0) {
    input_data.resize(taskData->inputs_count[0]);
    if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
      return false;
    }
    auto* source_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    std::copy(source_ptr, source_ptr + taskData->inputs_count[0], input_data.begin());
  }
  return true;
}

bool kudryashova_i_gather::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return ((taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 1) &&
            taskData->outputs_count[0] == 1 && taskData->outputs.size() == taskData->outputs_count.size() &&
            taskData->inputs_count[0] > 1 && taskData->inputs_count[0] % 2 == 0);
  }
  return true;
}

bool kudryashova_i_gather::TestMPITaskParallel::run() {
  internal_order_test();
  broadcast(world, segments.data(), processes, 0);
  if (world.rank() == 0) {
    size_t count = taskData->inputs_count[0];
    size_t halfSize = count / 2;
    firstHalf.resize(halfSize);
    secondHalf.resize(count - halfSize);
    std::copy(input_data.begin(), input_data.begin() + halfSize, firstHalf.begin());
    std::copy(input_data.begin() + halfSize, input_data.begin() + count, secondHalf.begin());
    int pointer = 0;
    for (int proc = 1; proc < world.size(); ++proc) {
      int proc_segment = segments[proc];
      world.send(proc, 0, firstHalf.data() + pointer, proc_segment);
      world.send(proc, 1, secondHalf.data() + pointer, proc_segment);
      pointer += proc_segment;
    }
  }
  if (world.rank() != 0) {
    local_input1_.resize(segments[world.rank()]);
    local_input2_.resize(segments[world.rank()]);
    world.recv(0, 0, local_input1_.data(), segments[world.rank()]);
    world.recv(0, 1, local_input2_.data(), segments[world.rank()]);
    local_result = std::inner_product(local_input1_.begin(), local_input1_.end(), local_input2_.begin(), 0);
  }
  std::vector<int> full_results;
  full_results.resize(segments[world.rank()]);
  gather(world, local_result, full_results, 0);
  if (world.rank() == 0) {
    if ((int)(taskData->inputs_count[0]) < world.size() || (world.size() == 1)) {
      result = std::inner_product(input_data.begin(), input_data.begin() + taskData->inputs_count[0] / 2,
                                  input_data.begin() + taskData->inputs_count[0] / 2, 0);
      return true;
    }
    result = std::accumulate(full_results.begin() + 1, full_results.end(), 0);
  }
  return true;
}

bool kudryashova_i_gather::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!taskData->outputs.empty()) {
      reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
    }
  }
  return true;
}
}  // namespace kudryashova_i_gather