// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/borisov_s_sum_of_rows/include/ops_mpi.hpp"

TEST(borisov_s_sum_of_rows, Test_Pipeline_Run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  size_t rows = 5000;
  size_t cols = 5000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(rows * cols, 1);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  auto testMpiTaskParallel = std::make_shared<borisov_s_sum_of_rows::SumOfRowsTaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < rows; ++i) {
      ASSERT_EQ(global_row_sums[i], 5000);
    }
  }
}

TEST(borisov_s_sum_of_rows, Test_Task_Run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_row_sums;

  size_t rows = 5000;
  size_t cols = 5000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(rows * cols, 1);
    global_row_sums.resize(rows, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_row_sums.data()));
    taskDataPar->outputs_count.push_back(global_row_sums.size());
  } else {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->outputs.emplace_back(nullptr);
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs_count.push_back(0);
  }

  auto testMpiTaskParallel = std::make_shared<borisov_s_sum_of_rows::SumOfRowsTaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < rows; ++i) {
      ASSERT_EQ(global_row_sums[i], 5000);
    }
  }
}