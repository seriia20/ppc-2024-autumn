#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/vavilov_v_min_elements_in_columns_of_matrix/include/ops_mpi.hpp"

std::vector<int> generate_rand_vec(int size, int lower_bound, int upper_bound) {
  std::vector<int> vec(size);
  for (auto& n : vec) {
    n = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return vec;
}

std::vector<std::vector<int>> generate_rand_matr(int rows, int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; i++) {
    matr[i] = generate_rand_vec(cols, -1000, 1000);
  }
  for (int j = 0; j < cols; j++) {
    int r_row = std::rand() % rows;
    matr[r_row][j] = INT_MIN;
  }
  return matr;
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, find_min_elem_in_col_400x500_matr) {
  boost::mpi::communicator world;
  const int rows = 400;
  const int cols = 500;

  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> min_col(cols, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = generate_rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(min_col.data()));
    taskDataPar->outputs_count.emplace_back(min_col.size());
  }

  vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_min(cols, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < cols; i++) {
      ASSERT_EQ(min_col[i], INT_MIN);
    }
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, find_min_elem_in_col_3000x3000_matr) {
  boost::mpi::communicator world;
  const int rows = 3000;
  const int cols = 3000;

  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> min_col(rows, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = generate_rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(min_col.data()));
    taskDataPar->outputs_count.emplace_back(min_col.size());
  }

  vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_min(rows, INT_MAX);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_min.data()));
    taskDataSeq->outputs_count.emplace_back(reference_min.size());

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < cols; i++) {
      ASSERT_EQ(min_col[i], INT_MIN);
    }
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, validation_input_empty_10x10_matr) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int rows = 10;
    const int cols = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matr = generate_rand_matr(rows, cols);

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> vec_res(cols, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
    taskDataSeq->outputs_count.emplace_back(vec_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, validation_output_empty_10x10_matr) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const int rows = 10;
    const int cols = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matr = generate_rand_matr(rows, cols);

    for (auto& row : matr) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> vec_res(cols, 0);
    taskDataSeq->outputs_count.emplace_back(vec_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, validation_find_min_elem_in_col_10x0_matr) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 10;
    const int cols = 0;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    std::vector<std::vector<int>> matr = generate_rand_matr(rows, cols);

    for (auto& row : matr) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> vec_res(cols, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
    taskDataSeq->outputs_count.emplace_back(vec_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, validation_fails_on_invalid_output_of_size) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 10;
    const int cols = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    std::vector<std::vector<int>> matr = generate_rand_matr(rows, cols);

    for (auto& row : matr) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    std::vector<int> vec_res(rows - 1, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
    taskDataSeq->outputs_count.emplace_back(vec_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, validation_empty_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 0;
    const int cols = 0;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    taskDataSeq->inputs_count = {rows, cols};
    std::vector<int> vec_res(cols, 0);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
    taskDataSeq->outputs_count.emplace_back(vec_res.size());

    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, find_min_elem_in_fixed_matrix) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    const int rows = 4;
    const int cols = 3;

    std::vector<std::vector<int>> fixed_matr = {{5, 3, 7}, {8, 1, 6}, {4, 9, 2}, {3, 0, 8}};

    std::vector<int> expected_min = {3, 0, 2};
    std::vector<int> result(cols, INT_MAX);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    for (auto& row : fixed_matr) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }

    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataSeq->outputs_count.emplace_back(result.size());

    vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (int i = 0; i < cols; i++) {
      ASSERT_EQ(result[i], expected_min[i]) << "Mismatch in column " << i;
    }
  }
}
