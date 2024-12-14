#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/Sdobnov_V_sum_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> generate_random_vector(int size, int lower_bound = 0, int upper_bound = 50) {
  std::vector<int> res(size);
  for (int i = 0; i < size; i++) {
    res[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return res;
}

std::vector<std::vector<int>> generate_random_matrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50) {
  std::vector<std::vector<int>> res(rows);
  for (int i = 0; i < rows; i++) {
    res[i] = generate_random_vector(columns, lower_bound, upper_bound);
  }
  return res;
  return std::vector<std::vector<int>>();
}

TEST(Sdobnov_V_sum_of_vector_elements_par, EmptyInput) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, EmptyOutput) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
  }
  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, EmptyMatrix) {
  boost::mpi::communicator world;
  int rows = 0;
  int columns = 0;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);

  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix1x1) {
  boost::mpi::communicator world;

  int rows = 1;
  int columns = 1;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix5x1) {
  boost::mpi::communicator world;

  int rows = 5;
  int columns = 1;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix10x10) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 10;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix100x100) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 100;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix100x10) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 10;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix10x100) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 100;
  int res = 0;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  test.validation();
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(respar, res);
  }
}
