#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/laganina_e_sum_values_by_columns_matrix/include/ops_mpi.hpp"

std::vector<int> laganina_e_sum_values_by_columns_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = (gen() % 100) - 49;
  }
  return vec;
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_2_2_matrix) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, 2, 1, 2};
  int n = 2;
  int m = 2;
  std::vector<int> empty_par(n, 0);
  std::vector<int> out = {2, 4};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(empty_par, empty_seq);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_500_300_matrix) {
  boost::mpi::communicator world;

  int n = 300;
  int m = 500;

  // Create data
  std::vector<int> in = laganina_e_sum_values_by_columns_matrix_mpi::getRandomVector(n * m);
  std::vector<int> empty_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(empty_par, empty_seq);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, partest1) {
  boost::mpi::communicator world;

  int n = 2;
  int m = 2;

  // Create data
  std::vector<int> in = {1, 2, 1, 2};
  std::vector<int> empty_par(n, 0);
  std::vector<int> out = {2, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(empty_par, empty_seq);
  }
}
TEST(laganina_e_sum_values_by_columns_matrix_mpi, partest2) {
  boost::mpi::communicator world;

  int n = 5000;
  int m = 3000;

  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> empty_par(n, 0);
  std::vector<int> out(n, m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(empty_par, empty_seq);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, partest3) {
  boost::mpi::communicator world;

  int n = 3000;
  int m = 5000;

  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> empty_par(n, 0);
  std::vector<int> out(n, m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_par.data()));
    taskDataPar->outputs_count.emplace_back(empty_par.size());
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> empty_seq(n, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty_seq.data()));
    taskDataSeq->outputs_count.emplace_back(empty_seq.size());

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(empty_par, empty_seq);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_14_13_20_19_13) {
  int n = 5;
  int m = 3;
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int32_t> out;
  std::vector<int32_t> res_par(n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = {2, 5, 6, 7, 4, 9, 4, 6, 7, 9, 3, 4, 8, 5, 0};
    out = {14, 13, 20, 19, 13};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> res_seq(n);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_par, out);
    ASSERT_EQ(res_seq, out);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_35_15_11_20_16_27) {
  int n = 6;
  int m = 3;
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int32_t> out;
  std::vector<int32_t> res_par(n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = {10, 7, 4, 8, 7, 9, 13, 4, 5, 7, 6, 9, 12, 4, 2, 5, 3, 9};
    out = {35, 15, 11, 20, 16, 27};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> res_seq(n);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_par, out);
    ASSERT_EQ(res_seq, out);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, Test_30_38_28_18_21) {
  int n = 5;
  int m = 4;
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int32_t> out;
  std::vector<int32_t> res_par(n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = {9, 5, 3, 9, 7, 9, 13, 4, 5, 7, 7, 9, 12, 4, 0, 5, 11, 9, 0, 7};
    out = {30, 38, 28, 18, 21};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_par.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }

  laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> res_seq(n);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(n);

    // Create Task
    laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_par, out);
    ASSERT_EQ(res_seq, out);
  }
}