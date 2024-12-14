#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/morozov_e_min_val_in_rows_matrix/include/ops_mpi.hpp"
std::vector<std::vector<int>> getRandomMatrix_(int n, int m) {
  int left = 0;
  int right = 10005;

  // Создаем матрицу
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));

  // Заполняем матрицу случайными значениями
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = left + std::rand() % (right - left + 1);
    }
  }
  for (int i = 0; i < n; ++i) {
    int m_ = std::rand() % m;
    matrix[i][m_] = -1;
  }
  return matrix;
}
TEST(morozov_e_min_val_in_rows_matrix_perf_test, test_pipeline_run_my) {
  boost::mpi::communicator world;
  const int n = 5000;
  const int m = 5000;
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
  std::vector<int> res(n);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix = getRandomMatrix_(n, m);
    for (int i = 0; i < n; ++i) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }
  auto testMpiTaskParallel = std::make_shared<morozov_e_min_val_in_rows_matrix::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(-1, res[i]);
    }
  }
}
TEST(morozov_e_min_val_in_rows_matrix_perf_test, test_task_run_my) {
  boost::mpi::communicator world;
  const int n = 450;
  const int m = 450;
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
  std::vector<int> res(n);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix = getRandomMatrix_(n, m);
    for (int i = 0; i < n; ++i) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }
  auto testMpiTaskParallel = std::make_shared<morozov_e_min_val_in_rows_matrix::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(-1, res[i]);
    }
  }
}