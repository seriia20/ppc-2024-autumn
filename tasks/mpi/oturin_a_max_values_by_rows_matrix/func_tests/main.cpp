#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/oturin_a_max_values_by_rows_matrix/include/ops_mpi.hpp"

std::vector<int> oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

// squarelike
TEST(oturin_a_max_values_by_rows_matrix_mpi_functest, Test_Max_1) {
  size_t n = 5;
  size_t m = 5;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world.rank() == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < global_max.size(); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

// rectangular
TEST(oturin_a_max_values_by_rows_matrix_mpi_functest, Test_Max_2) {
  size_t n = 10;
  size_t m = 15;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world.rank() == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < global_max.size(); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(oturin_a_max_values_by_rows_matrix_mpi_functest, Test_Max_3) {
  size_t n = 15;
  size_t m = 10;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world.rank() == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < global_max.size(); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(oturin_a_max_values_by_rows_matrix_mpi_functest, Test_Max_4) {
  size_t n = 1;
  size_t m = 15;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world.rank() == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < global_max.size(); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(oturin_a_max_values_by_rows_matrix_mpi_functest, Test_Max_5) {
  size_t n = 15;
  size_t m = 1;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world.rank() == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < global_max.size(); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}

TEST(oturin_a_max_values_by_rows_matrix_mpi_functest, Test_Max_EMPTY) {
  size_t n = 0;
  size_t m = 0;

  boost::mpi::communicator world;

  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  if (world.rank() == 0) {
    global_mat = oturin_a_max_values_by_rows_matrix_mpi::getRandomVector(n * m);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < global_max.size(); i++) {
      ASSERT_EQ(reference_max[i], global_max[i]);
    }
  }
}