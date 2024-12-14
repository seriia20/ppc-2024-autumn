// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/poroshin_v_count_min_val_row_matrix_all_reduce/include/ops_mpi.hpp"

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, test_pipeline_run) {
  int m = 1000;
  int n = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(m * n, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, test_task_run) {
  int m = 1000;
  int n = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(m * n, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_test_pipeline_run) {
  int m = 1000;
  int n = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(m * n, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_test_task_run) {
  int m = 1000;
  int n = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(m * n, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}