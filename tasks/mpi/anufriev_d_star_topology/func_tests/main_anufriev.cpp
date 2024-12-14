#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "mpi/anufriev_d_star_topology/include/ops_mpi_anufriev.hpp"

std::vector<int> createInputVector(size_t size, int initialValue = 0, int step = 1) {
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), initialValue);
  for (size_t i = 0; i < size; ++i) {
    vec[i] *= step;
  }
  return vec;
}

std::vector<int> calculateExpectedOutput(const std::vector<int>& input, int worldSize) {
  std::vector<int> output = input;
  size_t chunk_size = output.size() / worldSize;
  size_t remainder = output.size() % worldSize;

  for (int i = 0; i < worldSize; ++i) {
    size_t start_pos = i * chunk_size + std::min((size_t)i, remainder);
    size_t count = chunk_size + (static_cast<size_t>(i) < remainder ? 1 : 0);
    for (size_t j = 0; j < count; ++j) {
      output[start_pos + j] += i;
    }
  }

  return output;
}

std::vector<int> generate_random_vector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-1000, 1000);
  std::vector<int> result(size);
  std::generate(result.begin(), result.end(), [&]() { return dist(gen); });
  return result;
}

TEST(anufriev_d_star_topology, EmptyVectorTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  taskData->inputs_count.push_back(0);
  taskData->outputs_count.push_back(0);
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(0);
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }
  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    ASSERT_EQ(output_data.size(), static_cast<size_t>(0));
  }
}

TEST(anufriev_d_star_topology, SingleElementVectorTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(1);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, LargeVectorTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(1000);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, SmallDataFewerProcessesTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(world.size() - 1);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, LargeDataMoreProcessesTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(world.size() * 1000);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, NegativeAndPositiveNumbersTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = generate_random_vector(1000);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, IntMaxMinValuesTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = {INT_MAX, INT_MIN, -1, 0, 1};
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, UnevenDataDistributionTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    size_t total_size = world.size() * 1000;
    input_data.resize(total_size);
    std::fill(input_data.begin(), input_data.begin() + 10, 1);
    std::fill(input_data.begin() + 10, input_data.end(), 0);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    auto expected_output = calculateExpectedOutput(input_data, world.size());
    ASSERT_EQ(output_data, expected_output);
  }
}

TEST(anufriev_d_star_topology, MultipleRunsTest) {
  boost::mpi::communicator world;
  for (int run = 0; run < 3; ++run) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<int> input_data;
    std::vector<int> output_data;

    if (world.rank() == 0) {
      input_data = createInputVector(1000, run * 1000);
      taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
      taskData->inputs_count.push_back(input_data.size());

      output_data.resize(input_data.size());
      taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
      taskData->outputs_count.push_back(output_data.size());
    }

    auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
    ASSERT_TRUE(task->validation());
    ASSERT_TRUE(task->pre_processing());
    ASSERT_TRUE(task->run());
    ASSERT_TRUE(task->post_processing());

    if (world.rank() == 0) {
      auto expected_output = calculateExpectedOutput(input_data, world.size());
      ASSERT_EQ(output_data, expected_output);
    }
  }
}

TEST(anufriev_d_star_topology, DataPathTest) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data;
  std::vector<int> output_data;

  if (world.rank() == 0) {
    input_data = createInputVector(1000);
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskData->inputs_count.push_back(input_data.size());

    output_data.resize(input_data.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskData->outputs_count.push_back(output_data.size());
  }

  auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected_data_path;
    for (int i = 1; i < world.size(); ++i) {
      expected_data_path.push_back(i);
    }
    for (int i = 1; i < world.size(); ++i) {
      expected_data_path.push_back(i);
    }
    ASSERT_EQ(task->getDataPath(), expected_data_path);
  } else {
    std::vector<int> expected_data_path = {0, 0};
    ASSERT_EQ(task->getDataPath(), expected_data_path);
  }
}
